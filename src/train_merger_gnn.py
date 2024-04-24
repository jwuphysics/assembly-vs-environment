import schedulefree
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

class MangroveGCN(torch.nn.Module):
    """NOT USED -- SAGEConv net from Mangrove"""
    def __init__(self, n_in=9, hidden_channels=64, n_out=1, nlin=3):
        super(MangroveGCN, self).__init__()
        self.n_in = n_in
        self.n_hidden = hidden_channels
        self.n_out = n_out
                 
        self.node_enc = MangroveMLP(n_in, hidden_channels, layer_norm=True)
        
        self.conv1 = SAGEConv(hidden_channels, hidden_channels) 
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, hidden_channels)
        
        self.lin = nn.Linear(2*hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(normalized_shape=hidden_channels)
        self.lin_f = nn.Linear(hidden_channels, 2*n_out)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings 
        x = self.node_enc(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = torch.cat([global_max_pool(x, batch),global_add_pool(x, batch)], 1) 

        x = self.lin(x)
        x = x.relu()
        x = self.lin_f(self.norm(x))
        return x

class MangroveMLP(nn.Module):
    """NOT USED -- helper class for Mangrove GCN"""
    def __init__(self, n_in, n_out, hidden=64, nlayers=2, layer_norm=True):
        super(MangroveMLP, self).__init__()
        layers = [nn.Linear(n_in, hidden), nn.ReLU()]
        for i in range(nlayers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU()) 
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.Linear(hidden, n_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MultiSAGENet(torch.nn.Module):
    """A multi-layer GNN built using SAGEConv layers.

    Makes full graph-level predictions (i.e. one for each merger tree)
    """
    def __init__(
        self, 
        n_in=4, 
        n_hidden=16, 
        n_out=1, 
        n_layers=4, 
        bias=True,
        aggr=["max", "mean"]
    ):
        super(MultiSAGENet, self).__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.aggr = aggr
        self.bias = bias

        sage_convs = [SAGEConv(self.n_in, self.n_hidden, bias=self.bias, aggr=self.aggr)]                
        sage_convs += [SAGEConv(self.n_hidden, self.n_hidden, bias=self.bias, aggr=self.aggr) for _ in range(self.n_layers - 1)]
        self.convs = nn.ModuleList(sage_convs)
        
        self.readout = nn.Sequential(
            nn.Linear(3 * (self.n_layers * self.n_hidden + self.n_in), 4 * self.n_hidden, bias=self.bias),
            nn.SiLU(),
            nn.LayerNorm(4 * self.n_hidden),
            nn.Linear(4 * self.n_hidden, 2 * self.n_out, bias=self.bias)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        out = [
            torch.cat([
                global_mean_pool(x, data.batch),
                global_max_pool(x, data.batch), 
                global_add_pool(x, data.batch)
            ], axis=1)
        ]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.silu(x)

            out += [
                torch.cat([
                    global_mean_pool(x, data.batch),
                    global_max_pool(x, data.batch), 
                    global_add_pool(x, data.batch)
                ], axis=1)
            ]
              
        return self.readout(torch.cat(out, axis=1))
        
def train(dataloader, model, optimizer, device="cuda", augment_noise=True):
    """Train GNN model using Gaussian NLL loss."""
    model.train()

    loss_total = 0
    for data in (dataloader):
        if augment_noise: 
            # add random noise to halo mass and vmax
            data_node_features_scatter = 3e-4 * torch.randn_like(data.x[:, :-2]) * torch.std(data.x[:, :-2], dim=0)
            data.x[:, :-2] += data_node_features_scatter
            assert not torch.isnan(data.x).any() 

        data.to(device)

        optimizer.zero_grad()
        y_pred, logvar_pred = model(data).chunk(2, dim=1)
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()

        y_pred = y_pred.view(-1, model.n_out)
        logvar_pred = logvar_pred.mean()
        loss = 0.5 * (F.mse_loss(y_pred.view(-1), data.y) / 10**logvar_pred + logvar_pred)

        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return loss_total / len(dataloader)


def validate(dataloader, model, device="cuda"):
    model.eval()

    loss_total = 0

    y_preds = []
    y_trues = []
    root_subhalo_ids = []

    for data in (dataloader):
        with torch.no_grad():
            data.to(device)
            y_pred, logvar_pred = model(data).chunk(2, dim=1)
            y_pred = y_pred.view(-1, model.n_out)
            logvar_pred = logvar_pred.mean()
            loss = 0.5 * (F.mse_loss(y_pred.view(-1), data.y) / 10**logvar_pred + logvar_pred)

            loss_total += loss.item()
            y_preds += list(y_pred.detach().cpu().numpy())
            y_trues += list(data.y.detach().cpu().numpy())
            root_subhalo_ids += data.root_subhalo_id

    y_preds = np.concatenate(y_preds)
    y_trues = np.array(y_trues)
    root_subhalo_ids = np.array(root_subhalo_ids)

    return (
        loss_total / len(dataloader),
        y_preds,
        y_trues,
        root_subhalo_ids,
    )

def configure_optimizer(model, lr, wd,):
    """Only apply weight decay to weights, but not to other
    parameters like biases or LayerNorm. Based on minGPT version.
    """

    decay, no_decay = set(), set()
    yes_wd_modules = (nn.Linear, )
    no_wd_modules = (nn.LayerNorm, )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, yes_wd_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, no_wd_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=lr, 
    )

    return optimizer

def kfold_validate_trees():
    # use full trees: pruned versions don't do as well
    with open(f"{results_dir}/merger_trees.pkl", "rb") as f:
        trees = pickle.load(f)
    
    rng = np.random.RandomState(seed)
    rng.shuffle(trees)

    root_subhalo_ids = [t.root_subhalo_id for t in trees]

    # impose mass cut for convenience
    if only_centrals:
        trees = [tree for tree in trees if ((tree.is_central) & (tree.y > minimum_root_stellar_mass))]
    else: 
        trees = [tree for tree in trees if tree.y > minimum_root_stellar_mass]

    N = len(trees)
    
    for k in range(K):
        train_mask = np.full(N, True).astype(bool)
        train_mask[int(np.round(k) / K * N):int(np.round(k+1) / K * N)] = False
        valid_mask = ~train_mask
        
        train_loader = DataLoader(list(compress(trees, train_mask)), batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(list(compress(trees, valid_mask)), batch_size=batch_size, shuffle=False, pin_memory=True)
    
        model = MultiSAGENet(n_in=8, n_hidden=n_hidden, n_layers=n_layers, bias=bias, n_out=1).to(device)
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
                
            train_loss = train(train_loader, model, optimizer, device=device)
            valid_loss, p, y, root_subhalo_ids = validate(valid_loader, model, device=device)
        
            t1 = time.time()
        
            log_file.write(f" {epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    {np.sqrt(np.mean((p - y.flatten())**2)): >10.6f}    {t1-t0:.1f}s\n")
            log_file.flush()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        log_file.close()
        results = pd.DataFrame({
            "root_subhalo_ids": np.array(root_subhalo_ids),
            "p_envGNN_dmo": p,
            "log_Mstar": y,
        })
    
        results.to_csv(f"{results_dir}/predictions-mergertreeGNN-{k+1}of{K}.csv", index=False)
        
        model_fname = f"{results_dir}/models/merger_trees-mass_gt_{minimum_root_stellar_mass:g}-only_centrals_{int(only_centrals)}-{k+1}of{K}.pth"
        torch.save(model.state_dict(), model_fname)

def combine_validation_experiments(base_name="predictions-mergertreeGNN"):
    pm1 = pd.read_csv(f"{results_dir}/{base_name}-1of3.csv")
    pm1["k"] = 1
    pm2 = pd.read_csv(f"{results_dir}/{base_name}-2of3.csv")
    pm2["k"] = 2
    pm3 = pd.read_csv(f"{results_dir}/{base_name}-3of3.csv")
    pm3["k"] = 3
    preds_env = pd.concat([pm1, pm2, pm3])
    preds_env.to_csv(f"{results_dir}/{base_name}.csv", index=False)


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
        trees = [tree for tree in trees if ((tree.is_central) & (tree.y > minimum_root_stellar_mass))]
    else: 
        trees = [tree for tree in trees if tree.y > minimum_root_stellar_mass]
    
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
    
        model = MultiSAGENet(n_in=8, n_hidden=n_hidden, n_layers=n_layers, bias=bias, n_out=1).to(device)
        
        train_losses = []
        valid_losses = []
        
        # start off with very low learning rate (warm up phase)
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=wd, warmup_steps=20)

        log_file = open(f"{ROOT}/logs/train_env-gnn-residuals_merger_trees.log", "a")
        
        # what's the baseline error?
        valid_loss, p, y, root_subhalo_id = validate(valid_loader, model, device=device)
        log_file.write(f"Initial residual scatter (fold {k+1}): {y.std(): >10.6f}\n")
        
        log_file.write(f"Epoch    Train loss   Valid Loss       RSME      time\n")
        for epoch in range(n_residual_training_epochs):
    
            t0 = time.time()
                
            train_loss = train(train_loader, model, optimizer, device=device)
            valid_loss, p, y, root_subhalo_id = validate(valid_loader, model, device=device)
        
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
