import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
import numpy as np
import pandas as pd

from data import *
from loader import *
from model import EdgeInteractionGNN
from training_utils import (
    TrainingLogger,
    train_epoch_geometric,
    validate_geometric,
    combine_kfold_results,
    get_device,
    configure_optimizer
)

import pickle
import time

from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
results_dir = ROOT / "results"

device = get_device()

#######################
### HYPERPARAMETERS ###
#######################
n_epochs = 500
num_parts = 32

lr = 1e-2
wd = 1e-4

n_layers = 2
n_hidden = 32
n_latent = 8
n_unshared_layers = 4
aggr = "multi"

K = 3

def get_train_valid_indices(data, k, K=3, boxsize=75/0.6774, pad=3, epsilon=1e-10):
    """k must be between `range(0, K)`. 

    `boxsize` and `pad` are both in units of Mpc, and it is assumed that the 
    `data` object has attribute `pos` of shape (N_rows, 3) also in units of Mpc.

    `epsilon` is there so that the modular division doesn't cause the boolean
    logic to wrap around.
    """

    # use z coordinate for train-valid split
    train_1_mask = (
        (data.pos[:, 2]  > ((k) / K * boxsize + pad) % boxsize) 
        & (data.pos[:, 2] <= ((k + 1) / K * boxsize - epsilon) % boxsize)
    )

    train_2_mask = (
        (data.pos[:, 2]  > ((k + 1)/ K * boxsize) % boxsize) 
        & (data.pos[:, 2] <= ((k + 2) / K * boxsize - pad) % boxsize)
    )

    valid_mask = (
        (data.pos[:, 2] > ((k + 2) / K * boxsize) % boxsize)
        & (data.pos[:, 2] <= ((k + 3) / K * boxsize - epsilon) % boxsize)
    )

    # this is the weird pytorch way of doing `np.argwhere`
    train_indices = (train_1_mask  | train_2_mask).nonzero(as_tuple=True)[0] 

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    # ensure zero overlap
    assert (set(train_indices) & set(valid_indices)) == set()

    return train_indices, valid_indices


# Train function moved to training_utils.py


# Validate function moved to training_utils.py
    
# configure_optimizer function moved to training_utils.py

def kfold_validate_graphs():
    with open(f'{results_dir}/cosmic_graphs.pkl', 'rb') as f:
        data = pickle.load(f)

    node_features = data.x.shape[1]
    edge_features = data.edge_attr.shape[1]
    out_features = data.y.shape[1]

    for k in range(K):
       
        model = EdgeInteractionGNN(
            node_features=node_features, 
            edge_features=edge_features, 
            n_layers=n_layers, 
            hidden_channels=n_hidden,
            latent_channels=n_latent,
            n_unshared_layers=n_unshared_layers,
            n_out=out_features,
            aggr=(["sum", "max", "mean"] if aggr == "multi" else aggr)
        ).to("cuda")
    
        train_indices, valid_indices = get_train_valid_indices(data, k=k, K=K)
        
        train_data = ClusterData(
            data.subgraph(train_indices), 
            num_parts=num_parts, 
            recursive=False,
            log=False
        )
        train_loader = ClusterLoader(
            train_data,
            shuffle=True,
            batch_size=1,
        )
    
        valid_data = ClusterData(
            data.subgraph(valid_indices), 
            num_parts=num_parts // 2, 
            recursive=False,
            log=False
        )
        valid_loader = ClusterLoader(
            valid_data,
            shuffle=False, 
            batch_size=1,
        )
    
        optimizer = configure_optimizer(model, lr, wd)
    
        train_losses = []
        valid_losses = []

        log_file = open(f"{ROOT}/logs/train_env_graph.log", "a")
        log_file.write("Epoch    Train loss   Valid Loss      RSME        time \n")
        
        t0 = time.time()
        for epoch in range(n_epochs):
            
            if epoch == int(n_epochs * 0.25):
                optimizer = configure_optimizer(model, lr/5, wd)
            elif epoch == (n_epochs * 0.5):
                optimizer = configure_optimizer(model, lr/25, wd)
            elif epoch == (n_epochs * 0.75):
                optimizer = configure_optimizer(model, lr/125, wd)
        
            train_loss = train_epoch_geometric(train_loader, model, optimizer, device, augment=True, augment_strength=3e-3, augment_edges=True)
            valid_loss, p, y, *_  = validate_geometric(valid_loader, model, device, return_ids=True, return_centrals=True)
        
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
    
            if (epoch + 1) % 10 == 0:
                t1 = time.time()
                log_file.write(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((p - y.flatten())**2)): >10.6f}      {t1-t0:.2f}s\n")
                log_file.flush()
                t0 = time.time()

        log_file.close()
        
        # save predictions
        valid_loss, p, y, valid_idxs, _  = validate_geometric(valid_loader, model, device, return_ids=True, return_centrals=True)
        
        # helper function to select the validation indices for 1-d tensors in `data`
        # and return as numpy
        select_valid = lambda data_tensor: data_tensor[valid_indices].numpy().flatten()
    
        # since the valid ids get shuffled by ClusterData, we need to make sure that the subhalo_id column
        # is sorted, and then use that to rearrange the GNN predictions 
        # assert np.allclose(select_valid(data.subhalo_id), np.sort(select_valid(data.subhalo_id)))
        assert torch.allclose(torch.tensor(y[np.argsort(valid_idxs)]), data.y[valid_indices])
    
        results = pd.DataFrame({
            "valid_idxs": valid_idxs,
            "subhalo_id": select_valid(data.subhalo_id),
            "log_Mstar": select_valid(data.y),
            "p_envGNN_dmo": p[np.argsort(valid_idxs)],
            "log_Mhalo_dmo": select_valid(data.x[:, 0]),
            "log_Vmax_dmo": select_valid(data.x[:, 1]),
            "log_r50_dmo": select_valid(data.x[:, 3]),
            "log_MRmax_dmo": select_valid(data.x[:, 4]),
            "log_Vdisp_dmo": select_valid(data.x[:, 5]),
            "log_Vmaxrad_dmo": select_valid(data.x[:, 6]),
            "log_spin_dmo": select_valid(data.x[:, 7]),
            "x_dmo": select_valid(data.pos[:, 0]),
            "y_dmo": select_valid(data.pos[:, 1]),
            "z_dmo": select_valid(data.pos[:, 2]),
            "vx_dmo": select_valid(data.vel[:, 0]),
            "vy_dmo": select_valid(data.vel[:, 1]),
            "vz_dmo": select_valid(data.vel[:, 2]),
            "is_central": select_valid(data.is_central).astype(int),
            "log_Mhalo_hydro": select_valid(data.x_hydro[:, 0]),
            "log_Vmax_hydro": select_valid(data.x_hydro[:, 1]),
            "x_hydro": select_valid(data.pos_hydro[:, 0]),
            "y_hydro": select_valid(data.pos_hydro[:, 1]),
            "z_hydro": select_valid(data.pos_hydro[:, 2]),
            "vx_hydro": select_valid(data.vel_hydro[:, 0]),
            "vy_hydro": select_valid(data.vel_hydro[:, 1]),
            "vz_hydro": select_valid(data.vel_hydro[:, 2]),
            "Rstar_50": select_valid(data.halfmassradius),
            "overdensity": select_valid(data.overdensity),
        })
    
        results.to_csv(f"{results_dir}/predictions-envGNN-{k+1}of{K}.csv", index=False)


def combine_validation_experiments():
    """Combine k-fold validation results - wrapper for utility function."""
    return combine_kfold_results(results_dir, "predictions-envGNN", K)


if __name__ == "__main__":
    # run training_loop
    prepare_all_data()
    kfold_validate_graphs()
    combine_validation_experiments()