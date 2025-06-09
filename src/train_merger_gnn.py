import schedulefree
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, SAGEConv
import numpy as np
import pandas as pd

from data import *
from loader import *
from model import MultiSAGENet
from training_utils import (
    TrainingLogger, 
    train_epoch_geometric, 
    validate_geometric, 
    combine_kfold_results,
    get_device,
    configure_optimizer
)

import pickle
import random
from itertools import compress
import time

from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
results_dir = ROOT / "results"

device = get_device()


#######################
### HYPERPARAMETERS ###
#######################
minimum_root_stellar_mass = 8.5
only_centrals = True

n_epochs = 500
batch_size = 128

lr = 1e-2
wd = 1e-4

n_layers = 8
n_hidden = 8
bias=False

seed = 42
K = 3

# Unused classes removed - they were not being used in the training pipeline

# MultiSAGENet is now imported from model.py to avoid duplication
        
# Train function moved to training_utils.py


# Validate function moved to training_utils.py

# configure_optimizer function moved to training_utils.py

def kfold_validate_trees():
    # use full trees: pruned versions don't do as well
    with open(f"{results_dir}/merger_trees.pkl", "rb") as f:
        trees = pickle.load(f)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(trees)

    root_subhalo_ids = [t.root_subhalo_id for t in trees]

    # impose mass cut for convenience
    if only_centrals:
        trees = [tree for tree in trees if ((tree.is_central) & (tree.y[0, 0] > minimum_root_stellar_mass))]
    else: 
        trees = [tree for tree in trees if tree.y[0, 0] > minimum_root_stellar_mass]

    N = len(trees)
    
    for k in range(K):
        train_mask = np.full(N, True).astype(bool)
        train_mask[int(np.round(k) / K * N):int(np.round(k+1) / K * N)] = False
        valid_mask = ~train_mask
        
        train_loader = DataLoader(list(compress(trees, train_mask)), batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(list(compress(trees, valid_mask)), batch_size=batch_size, shuffle=False, pin_memory=True)
    
        model = MultiSAGENet(n_in=8, n_hidden=n_hidden, n_layers=n_layers, bias=bias, n_out=2).to(device)
        # model = GCN(n_in=8, hidden_channels=128, n_out=1).to(device)
        
        train_losses = []
        valid_losses = []
        
        # start off with very low learning rate (warm up phase)
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=wd, warmup_steps=20)
        # optimizer = configure_optimizer(model, lr, wd)

        log_file = open(f"{ROOT}/logs/train_merger_trees.log", "a")
        
        log_file.write(f"Epoch    Train loss   Valid Loss       RSME      time\n")
        for epoch in range(n_epochs):
    
            t0 = time.time()
                
            train_loss = train_epoch_geometric(train_loader, model, optimizer, device, augment_strength=3e-4, augment_edges=False)
            valid_loss, p, y, root_subhalo_ids = validate_geometric(valid_loader, model, device, return_ids=True, return_centrals=False)
        
            t1 = time.time()
        
            log_file.write(f" {epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((p - y.flatten())**2)): >10.6f}    {t1-t0:.1f}s\n")
            log_file.flush()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        log_file.close()
        results = pd.DataFrame({
            "fold": k,
            "index": np.array(root_subhalo_ids),
            "root_subhalo_ids": np.array(root_subhalo_ids),
            "p_mergertreeGNN_Mstar": p[:, 0],
            "p_mergertreeGNN_Mgas": p[:, 1],
            "target_Mstar": y[:, 0],
            "target_Mgas": y[:, 1],
        })
    
        results.to_csv(f"{results_dir}/predictions-mergertreeGNN-{k+1}of{K}.csv", index=False)
        
        model_fname = f"{results_dir}/models/merger_trees-mass_gt_{minimum_root_stellar_mass:g}-only_centrals_{int(only_centrals)}-{k+1}of{K}.pth"
        torch.save(model.state_dict(), model_fname)

def combine_validation_experiments(base_name="predictions-mergertreeGNN"):
    """Combine k-fold validation results - wrapper for utility function."""
    return combine_kfold_results(results_dir, base_name, K)


def predict_env_gnn_residuals(n_residual_training_epochs=100):
    """Assuming an environmental graph has already been trained, now compute residuals
    from those env predictions and regress them using a merger tree GNN.

    Uses the same k-fold split as the env GNN trained on.
    """
    try:
        preds_env = pd.read_csv(f"{results_dir}/predictions-environment_GNN.csv")
    except FileNotFoundError:
        raise("Train the environment graph first and save it in `results_dir`/predictions-environment_GNN.csv!")

    
    preds_env["residual_envGNN"] = preds_env.p_envGNN_dmo - preds_env.log_Mstar
    preds_env.set_index("subhalo_id", inplace=True)

    with open(f"{results_dir}/merger_trees.pkl", "rb") as f:
        trees = pickle.load(f)
    
    # impose mass cut for convenience
    if only_centrals:
        trees = [tree for tree in trees if ((tree.is_central) & (tree.y[0, 0] > minimum_root_stellar_mass))]
    else: 
        trees = [tree for tree in trees if tree.y[0, 0] > minimum_root_stellar_mass]
    
    for tree in trees:
        tree.log_Mstar = tree.y # hang on to this
        tree.y = torch.tensor([preds_env.loc[tree.root_subhalo_id].residual_envGNN], dtype=torch.float32)
        tree.k = preds_env.loc[tree.root_subhalo_id].k.astype(int)
    
    # actual training part
    for k in range(K):
        # use same train/test split as env graph
        train_mask = np.array([t.k != (k + 1) for t in trees])
        valid_mask = ~train_mask
        
        train_loader = DataLoader(list(compress(trees, train_mask)), batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(list(compress(trees, valid_mask)), batch_size=batch_size, shuffle=False, pin_memory=True)
    
        model = MultiSAGENet(n_in=8, n_hidden=n_hidden, n_layers=n_layers, bias=bias, n_out=2).to(device)
        
        train_losses = []
        valid_losses = []
        
        # start off with very low learning rate (warm up phase)
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=wd, warmup_steps=20)

        log_file = open(f"{ROOT}/logs/train_env-gnn-residuals_merger_trees.log", "a")
        
        # what's the baseline error?
        valid_loss, p, y, root_subhalo_id = validate_geometric(valid_loader, model, device, return_ids=True, return_centrals=False)
        log_file.write(f"Initial residual scatter (fold {k+1}): {y.std(): >10.6f}\n")
        
        log_file.write(f"Epoch    Train loss   Valid Loss       RSME      time\n")
        for epoch in range(n_residual_training_epochs):
    
            t0 = time.time()
                
            train_loss = train_epoch_geometric(train_loader, model, optimizer, device, augment_strength=3e-4, augment_edges=False)
            valid_loss, p, y, root_subhalo_id = validate_geometric(valid_loader, model, device, return_ids=True, return_centrals=False)
        
            t1 = time.time()
        
            log_file.write(f" {epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((p - y.flatten())**2)): >10.6f}    {t1-t0:.1f}s\n")
            log_file.flush()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        results = pd.DataFrame({
            "root_subhalo_id": np.array(root_subhalo_id),
            "p_mergertree_on_residual": p,
            "residual_envGNN": y,
        })

        results.to_csv(f"{results_dir}/predictions_envGNN_residuals-mergertreeGNN-{k+1}of{K}.csv", index=False)

        model_fname = f"{results_dir}/models/train_env-gnn-residuals_merger_trees-{k+1}of{K}.pth"
        torch.save(model.state_dict(), model_fname)
        

if __name__ == "__main__":
    # train and cross-validate trees
    # kfold_validate_trees()
    # combine_validation_experiments()

    # assuming that the env GNN has already been trained, now train a merger tree to regress pred_envGNN_logMstar - true_logMstar
    # predict_env_gnn_residuals(n_residual_training_epochs=100)
    combine_validation_experiments(base_name="predictions_envGNN_residuals-mergertreeGNN")
