import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data import *
from loader import *
from training_utils import (
    TrainingLogger,
    train_epoch_tensor,
    validate_tensor,
    combine_kfold_results,
    get_device
)

import pickle
import time
from sklearn.model_selection import KFold

from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
results_dir = ROOT / "results"

device = get_device()

#######################
### HYPERPARAMETERS ###
#######################
seed = 42

n_epochs = 200
batch_size = 1024

lr = 1e-3
wd = 1e-5

n_hidden = 128

K = 3

class SubhaloDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Train function moved to training_utils.py


# Validate function moved to training_utils.py


def predict_mlp():
    """Main training loop for MLP estimates (only halo properties)"""
    with open(f'{results_dir}/cosmic_graphs.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data.x
    y = data.y
    
    node_features = X.shape[1]
    out_features = y.shape[1]
    
    N = len(X)
    
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    
    for k, (train_index, valid_index) in enumerate(kf.split(X, y)):
    
        train_dataset = SubhaloDataset(X[train_index], y[train_index])
        valid_dataset = SubhaloDataset(X[valid_index], y[valid_index])
    
        subhalo_ids = np.array(data.subhalo_id[valid_index])
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    
        model = nn.Sequential(
            nn.Linear(node_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 2* out_features)
        ).to(device)
        
        train_losses = []
        valid_losses = []
            
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        
        log_file = open(f"{ROOT}/logs/train_mlp.centrals.log", "a")
        log_file.write(f"Epoch    Train loss   Valid Loss      RSME \n")
        t0 = time.time()
        for epoch in range(n_epochs):
        
    
            train_loss = train_epoch_tensor(train_loader, model, optimizer, device)
            valid_loss, preds, trues = validate_tensor(valid_loader, model, device)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if (epoch + 1) % 10 == 0:
                t1 = time.time()
                log_file.write(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((preds - trues.flatten())**2)): >10.6f}      {t1-t0:.2f}s\n")
                log_file.flush()
                t0 = time.time()
            
        results = pd.DataFrame({
            "subhalo_id": np.array(subhalo_ids).reshape(-1),
            "p_haloonly_dmo": np.array(preds).reshape(-1),
            "log_Mstar": np.array(trues).reshape(-1),
        })
        log_file.close()
    
        results.to_csv(f"{results_dir}/predictions-halo_only-{k+1}of{K}.csv", index=False)

def combine_validation_experiments():
    """Combine k-fold validation results - wrapper for utility function."""
    return combine_kfold_results(results_dir, "predictions-halo_only", K)

if __name__ == "__main__":
    #predict_mlp()
    combine_validation_experiments()
