"""
Training utilities for assembly vs environment research.

This module consolidates common training functions and utilities used across
different model training scripts to reduce code duplication and improve maintainability.
"""

import time
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pathlib import Path
try:
    from .config import RANDOM_SEED, DEVICE_PREFERENCE
except ImportError:
    try:
        from config import RANDOM_SEED, DEVICE_PREFERENCE
    except ImportError:
        # Fallback values if config not available
        RANDOM_SEED = 42
        DEVICE_PREFERENCE = "cuda"


def configure_optimizer(model: nn.Module, lr: float, wd: float) -> torch.optim.AdamW:
    """Configure AdamW optimizer with selective weight decay.
    
    Only apply weight decay to weights, but not to other parameters like 
    biases or LayerNorm parameters. Based on minGPT implementation.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        wd: Weight decay
        
    Returns:
        Configured AdamW optimizer
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

    optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    return optimizer


def gaussian_nll_loss(y_pred: torch.Tensor, y_true: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian negative log-likelihood loss.
    
    Args:
        y_pred: Model predictions
        y_true: Ground truth values  
        logvar: Log variance predictions
        
    Returns:
        Gaussian NLL loss
    """
    # Create mask for valid targets (exclude negative values)
    valid_mask = (y_true < 0.)
    
    # If no valid targets, return zero loss
    if not valid_mask.any():
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    # Apply mask to predictions and targets
    y_pred_masked = y_pred[valid_mask]
    y_true_masked = y_true[valid_mask]
    
    # Compute loss only on valid samples
    return 0.5 * (F.mse_loss(y_pred_masked, y_true_masked) / 10**logvar + logvar)


def apply_data_augmentation(data, augment_strength: float = 3e-3, augment_edges: bool = True):
    """Apply data augmentation to graph data.
    
    Args:
        data: PyTorch Geometric data object
        augment_strength: Strength of noise to add
        augment_edges: Whether to augment edge features
    """
    # Augment node features (excluding last 2 features which might be special)
    if hasattr(data, 'x') and data.x.size(1) > 2:
        node_noise = augment_strength * torch.randn_like(data.x[:, :-2]) * torch.std(data.x[:, :-2], dim=0)
        data.x[:, :-2] += node_noise
        assert not torch.isnan(data.x).any(), "NaN values found in node features after augmentation"
    
    # Augment edge features if requested and available
    if augment_edges and hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_noise = augment_strength * torch.randn_like(data.edge_attr) * torch.std(data.edge_attr, dim=0)
        data.edge_attr += edge_noise
        assert not torch.isnan(data.edge_attr).any(), "NaN values found in edge features after augmentation"


def train_epoch_geometric(
    dataloader: GeometricDataLoader, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: str,
    augment: bool = True,
    augment_strength: float = 3e-3,
    augment_edges: bool = True
) -> float:
    """Train one epoch for PyTorch Geometric models.
    
    Args:
        dataloader: Data loader for training data
        model: Model to train
        optimizer: Optimizer
        device: Device to train on
        augment: Whether to apply data augmentation
        augment_strength: Strength of augmentation noise
        augment_edges: Whether to augment edge features
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    loss_total = 0
    
    for data in dataloader:
        if augment:
            apply_data_augmentation(data, augment_strength, augment_edges)
            
        data = data.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        y_pred, logvar_pred = output.chunk(2, dim=1)
        
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        
        y_pred = y_pred.view(-1, model.n_out if hasattr(model, 'n_out') else 2)
        logvar_pred = logvar_pred.mean()
        
        loss = gaussian_nll_loss(y_pred, data.y, logvar_pred)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        
    return loss_total / len(dataloader)


def train_epoch_tensor(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """Train one epoch for standard tensor-based models.
    
    Args:
        dataloader: Data loader for training data (X, y tuples)
        model: Model to train
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    loss_total = 0
    
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        y_pred, logvar_pred = output.chunk(2, dim=1)
        
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        
        y_pred = y_pred.view(-1, y.shape[1] if len(y.shape) > 1 else 1)
        logvar_pred = logvar_pred.mean()
        
        loss = gaussian_nll_loss(y_pred, y, logvar_pred)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        
    return loss_total / len(dataloader)


def validate_geometric(
    dataloader: GeometricDataLoader,
    model: nn.Module,
    device: str,
    return_ids: bool = True,
    return_centrals: bool = False
) -> Tuple[float, np.ndarray, np.ndarray, Any]:
    """Validate PyTorch Geometric model.
    
    Args:
        dataloader: Validation data loader
        model: Model to validate
        device: Device to validate on
        return_ids: Whether to return subhalo IDs
        return_centrals: Whether to return is_central flags
        
    Returns:
        Tuple of (loss, predictions, targets, optional_additional_data)
    """
    model.eval()
    loss_total = 0
    y_preds = []
    y_trues = []
    additional_data = []
    
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            output = model(data)
            y_pred, logvar_pred = output.chunk(2, dim=1)
            
            y_pred = y_pred.view(-1, model.n_out if hasattr(model, 'n_out') else 2)
            logvar_pred = logvar_pred.mean()
            
            loss = gaussian_nll_loss(y_pred, data.y, logvar_pred)
            loss_total += loss.item()
            
            y_preds.append(y_pred.detach().cpu().numpy())
            y_trues.append(data.y.detach().cpu().numpy())
            
            # Collect additional data if requested
            batch_additional = {}
            if return_ids and hasattr(data, 'idx'):
                batch_additional['ids'] = data.idx.detach().cpu().numpy()
            elif return_ids and hasattr(data, 'subhalo_id'):
                batch_additional['ids'] = data.subhalo_id.detach().cpu().numpy()
            elif return_ids and hasattr(data, 'root_subhalo_id'):
                batch_additional['ids'] = data.root_subhalo_id
                
            if return_centrals and hasattr(data, 'is_central'):
                batch_additional['is_central'] = data.is_central.detach().cpu().numpy()
                
            additional_data.append(batch_additional)
    
    y_preds = np.concatenate(y_preds, axis=0) if y_preds else np.array([])
    y_trues = np.concatenate(y_trues, axis=0) if len(y_trues) > 0 else np.array([])
    
    # Combine additional data
    combined_additional = {}
    if additional_data:
        for key in additional_data[0].keys():
            if key == 'ids' and not isinstance(additional_data[0][key], (list, np.ndarray)):
                # Handle case where IDs are not arrays (like root_subhalo_id)
                combined_additional[key] = [batch[key] for batch in additional_data]
            else:
                combined_additional[key] = np.concatenate([batch[key] for batch in additional_data])
    
    result = [loss_total / len(dataloader), y_preds, y_trues]
    if return_ids:
        result.append(combined_additional.get('ids', []))
    if return_centrals:
        result.append(combined_additional.get('is_central', []))
        
    return tuple(result)


def validate_tensor(
    dataloader: DataLoader,
    model: nn.Module,
    device: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate tensor-based model.
    
    Args:
        dataloader: Validation data loader
        model: Model to validate
        device: Device to validate on
        
    Returns:
        Tuple of (loss, predictions, targets)
    """
    model.eval()
    loss_total = 0
    y_preds = []
    y_trues = []
    
    for X, y in dataloader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            
            output = model(X)
            y_pred, logvar_pred = output.chunk(2, dim=1)
            
            y_pred = y_pred.view(-1, y.shape[1] if len(y.shape) > 1 else 1)
            logvar_pred = logvar_pred.mean()
            
            loss = gaussian_nll_loss(y_pred, y, logvar_pred)
            loss_total += loss.item()
            
            y_preds.append(y_pred.detach().cpu().numpy())
            y_trues.append(y.detach().cpu().numpy())
    
    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    
    return loss_total / len(dataloader), y_preds, y_trues


class TrainingLogger:
    """Training logger for consistent logging across experiments."""
    
    def __init__(self, log_file_path: Union[str, Path], experiment_name: str = ""):
        self.log_file_path = Path(log_file_path)
        self.experiment_name = experiment_name
        self.start_time = None
        
        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def write_header(self):
        """Write header for training log."""
        with open(self.log_file_path, "a") as f:
            if self.experiment_name:
                f.write(f"\n=== {self.experiment_name} ===\n")
            f.write("Epoch    Train loss   Valid Loss      RMSE        time\n")
        
    def log_epoch(self, epoch: int, train_loss: float, valid_loss: float, 
                  predictions: np.ndarray, targets: np.ndarray, 
                  interval: int = 10, flush: bool = True):
        """Log training metrics for an epoch."""
        if (epoch + 1) % interval == 0:
            if self.start_time is None:
                self.start_time = time.time()
                
            current_time = time.time()
            rmse = np.sqrt(np.mean((predictions - targets)**2))
            
            with open(self.log_file_path, "a") as f:
                f.write(f"{epoch + 1: >4d}    {train_loss: >9.5f}    {valid_loss: >9.5f}    "
                       f"{rmse: >10.6f}    {current_time - self.start_time:.1f}s\n")
                if flush:
                    f.flush()
                    
            self.start_time = current_time
            
    def close(self):
        """Close the logger (placeholder for future cleanup if needed)."""
        pass


def combine_kfold_results(results_dir: Path, base_name: str, k_folds: int = 3) -> pd.DataFrame:
    """Combine k-fold cross-validation results into a single DataFrame.
    
    Args:
        results_dir: Directory containing result files
        base_name: Base name of result files (e.g., "predictions-envGNN")
        k_folds: Number of folds
        
    Returns:
        Combined DataFrame with all k-fold results
    """
    dfs = []
    for k in range(1, k_folds + 1):
        file_path = results_dir / f"{base_name}-{k}of{k_folds}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["k"] = k
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found, skipping...")
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        output_path = results_dir / f"{base_name}.csv"
        combined.to_csv(output_path, index=False)
        print(f"Combined results saved to {output_path}")
        return combined
    else:
        raise FileNotFoundError(f"No k-fold result files found for {base_name}")


def get_device(prefer_cuda: Optional[bool] = None) -> str:
    """Get the appropriate device for training.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available. If None, uses config default.
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    if prefer_cuda is None:
        prefer_cuda = (DEVICE_PREFERENCE == "cuda")
        
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_spatial_train_valid_indices(data, k: int, K: int = 3, boxsize: float = 75/0.6774, 
                                   pad: float = 3, epsilon: float = 1e-10):
    """Create spatial train/validation indices using z-coordinate splits.
    
    This creates spatially separated train/validation sets by dividing the simulation
    box along the z-axis. Each fold uses 1/K of the box for validation and the rest
    for training (with padding to avoid boundary effects).
    
    Args:
        data: PyTorch Geometric data object with pos attribute
        k: Fold index (0 to K-1)
        K: Total number of folds
        boxsize: Simulation box size in Mpc
        pad: Padding between train/valid regions in Mpc
        epsilon: Small value to avoid boundary issues
        
    Returns:
        Tuple of (train_indices, valid_indices) as torch tensors
    """
    z_coords = data.pos[:, 2]
    
    # Calculate validation region boundaries
    valid_start = (k / K * boxsize) % boxsize
    valid_end = ((k + 1) / K * boxsize) % boxsize
    
    # Handle wrap-around case
    if valid_start > valid_end:  # Wraps around the boundary
        valid_mask = (z_coords >= valid_start) | (z_coords <= valid_end)
    else:
        valid_mask = (z_coords >= valid_start) & (z_coords <= valid_end)
    
    # Create training region with padding
    train_start = ((k + 1) / K * boxsize + pad) % boxsize
    train_end = (k / K * boxsize - pad) % boxsize
    
    # Handle wrap-around for training region
    if train_start > train_end:  # Wraps around the boundary
        train_mask = (z_coords >= train_start) | (z_coords <= train_end)
    else:
        train_mask = (z_coords >= train_start) & (z_coords <= train_end)
    
    # Get indices
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    
    # Ensure zero overlap
    overlap = set(train_indices.tolist()) & set(valid_indices.tolist())
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices between train and validation"
    
    print(f"Fold {k}/{K}: Train={len(train_indices)}, Valid={len(valid_indices)}")
    
    return train_indices, valid_indices


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)