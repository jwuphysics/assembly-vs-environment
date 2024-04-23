import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, SAGEConv, DirGNNConv
from torch_geometric.utils import remove_self_loops

from data import *
from loader import *
from model import *

import matplotlib.pyplot as plt
import pickle
import random
from itertools import compress
from sklearn.model_selection import KFold
from scipy.stats import binned_statistic, median_abs_deviation
import time

from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
results_dir = ROOT / "results"

device = "cuda"

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

def train(dataloader, model, optimizer, device):
    """Train GNN model using Gaussian NLL loss."""
    model.train()

    loss_total = 0
    for X, y in (dataloader):

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred, logvar_pred = model(X).chunk(2, dim=1)
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()

        y_pred = y_pred.view(-1, 1)
        logvar_pred = logvar_pred.mean()
        loss = 0.5 * (F.mse_loss(y_pred, y) / 10**logvar_pred + logvar_pred)

        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return loss_total / len(dataloader)


def validate(dataloader, model, device):
    model.eval()

    loss_total = 0

    y_preds = []
    y_trues = []
    subhalo_ids = []

    for X, y in (dataloader):
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            y_pred, logvar_pred = model(X).chunk(2, dim=1)
            y_pred = y_pred.view(-1, 1)
            logvar_pred = logvar_pred.mean()
            loss = 0.5 * (F.mse_loss(y_pred, y) / 10**logvar_pred + logvar_pred)

            loss_total += loss.item()
            
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(y.detach().cpu().numpy())

    y_preds = np.concatenate(y_preds)
    y_trues = np.array(y_trues)

    return (
        loss_total / len(dataloader),
        y_preds,
        y_trues
    )


def main():
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
        
        print(f"Epoch    Train loss   Valid Loss      RSME ")
        t0 = time.time()
        for epoch in range(n_epochs):
        
    
            train_loss = train(train_loader, model, optimizer, device=device)
            valid_loss, preds, trues = validate(valid_loader, model, device=device)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if (epoch + 1) % 10 == 0:
                t1 = time.time()
                print(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((preds - trues.flatten())**2)): >10.6f}      {t1-t0:.2f}s")
                t0 = time.time()
            
        results = pd.DataFrame({
            "subhalo_id": np.array(subhalo_ids).reshape(-1),
            "p_haloonly_dmo": np.array(preds).reshape(-1),
            "log_Mstar": np.array(trues).reshape(-1),
        })
    
        results.to_csv(f"{results_dir}/predictions-halo_only-{k+1}of{K}.csv", index=False)