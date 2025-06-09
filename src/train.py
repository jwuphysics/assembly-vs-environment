import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from scipy.stats import binned_statistic, median_abs_deviation
from sklearn.model_selection import KFold

from model import SAGEGraphConvNet, EdgeInteractionGNN
from data import prepare_subhalos, load_trees, make_merger_tree_graphs 

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, optimizer, device="cuda"):
    model.train()
    loss_total = 0
    for data in dataloader:
        data.to(device)       
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.mse_loss(y_pred, data.y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(dataloader)

def validate(dataloader, model, device="cuda"):
    model.eval()
    loss_total = 0
    y_preds = []
    y_trues = []
    for data in dataloader:
        data.to(device)
        with torch.no_grad():
            y_pred = model(data)
            loss = F.mse_loss(y_pred, data.y)
            loss_total += loss.item()
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(data.y.detach().cpu().numpy())
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    return loss_total / len(dataloader), y_preds, y_trues

def train_merger_tree(model=None, n_epochs=100, K=5):
    with open("merger_trees.pkl", "rb") as f:
        trees = pickle.load(f)

    
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    for k, (train_idx, valid_idx) in enumerate(kf.split(trees)):
        train_dataset = [trees[i] for i in train_idx]
        valid_dataset = [trees[i] for i in valid_idx]
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        if model is None:
            model = SAGEGraphConvNet(n_in=9).to(device)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(n_epochs):
            train_loss = train(train_loader, model, optimizer, device=device)
            valid_loss, _, _ = validate(valid_loader, model, device=device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.5f}, Valid Loss = {valid_loss:.5f}")

def train_point_cloud(model=None, n_epochs=100, K=5):
    with open("point_clouds.pkl", "rb") as f:
        point_clouds = pickle.load(f)

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    for k, (train_idx, valid_idx) in enumerate(kf.split(point_clouds)):
        train_dataset = [point_clouds[i] for i in train_idx]
        valid_dataset = [point_clouds[i] for i in valid_idx]
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

        if model is None:
            model = EdgeInteractionGNN(
                n_layers=3, 
                node_features=8, 
                edge_features=6, 
                hidden_channels=43, 
                aggr=["sum", "max", "mean"], 
                latent_channels=64, 
                n_out=1, 
                n_unshared_layers=4, 
                act_fn=torch.nn.SiLU()
            ).to(device)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        for epoch in range(n_epochs):
            train_loss = train(train_loader, model, optimizer, device=device)
            valid_loss, _, _ = validate(valid_loader, model, device=device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.5f}, Valid Loss = {valid_loss:.5f}")

if __name__ == "__main__":
    train_merger_tree()