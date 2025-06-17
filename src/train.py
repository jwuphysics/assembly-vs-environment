"""
Training wrapper with experiment tracking for reproducible comparisons.

This module provides high-level training functions that automatically:
1. Use consistent train/validation splits
2. Record all training artifacts
3. Save detailed predictions for analysis
4. Enable cross-model comparisons
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as GeometricDataLoader, ClusterData, ClusterLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import time

try:
    from .config import get_model_config, RESULTS_DIR, K_FOLDS, RANDOM_SEED
except ImportError:
    from config import get_model_config, RESULTS_DIR, K_FOLDS, RANDOM_SEED
try:
    from .model import MultiSAGENet, EdgeInteractionGNN
    from .utils import (
        TrainingLogger, train_epoch_geometric, validate_geometric,
        train_epoch_tensor, validate_tensor, configure_optimizer, get_device
    )
    from .experiment_tracker import ExperimentTracker
    from .loader import prepare_all_data
except ImportError:
    from model import MultiSAGENet, EdgeInteractionGNN
    from utils import (
        TrainingLogger, train_epoch_geometric, validate_geometric,
        train_epoch_tensor, validate_tensor, configure_optimizer, get_device
    )
    from experiment_tracker import ExperimentTracker
    from loader import prepare_all_data


def train_env_gnn_tracked(experiment_name: str, fold: int = None) -> Dict[str, Any]:
    """Train Environment GNN with full tracking.
    
    Args:
        experiment_name: Name of the experiment for tracking
        fold: Specific fold to train (if None, trains all folds)
        
    Returns:
        Dictionary with training results and file paths
    """
    # Set up experiment tracking
    tracker = ExperimentTracker(experiment_name)
    config = get_model_config('env_gnn')
    device = get_device()
    
    # Ensure data is prepared
    prepare_all_data()
    
    # Load data and create consistent splits
    with open(RESULTS_DIR / "cosmic_graphs.pkl", 'rb') as f:
        data = pickle.load(f)
    
    splits = tracker.create_consistent_splits("cosmic_graph", "spatial")
    
    # Determine which folds to train
    folds_to_train = [fold] if fold is not None else list(range(K_FOLDS))
    
    results = {}
    
    for k in folds_to_train:
        print(f"\n=== Training Environment GNN - Fold {k+1}/{K_FOLDS} ===")
        
        # Get train/validation indices
        train_indices = torch.tensor(splits[k]['train'])
        valid_indices = torch.tensor(splits[k]['valid'])
        
        # Create data loaders with ClusterData for efficiency
        train_data = ClusterData(
            data.subgraph(train_indices),
            num_parts=config['num_parts'],
            recursive=False,
            log=False
        )
        train_loader = ClusterLoader(train_data, shuffle=True, batch_size=1)
        
        valid_data = ClusterData(
            data.subgraph(valid_indices),
            num_parts=config['num_parts'] // 2,
            recursive=False,
            log=False
        )
        valid_loader = ClusterLoader(valid_data, shuffle=False, batch_size=1)
        
        # Initialize model
        model = EdgeInteractionGNN(
            node_features=data.x.shape[1],
            edge_features=data.edge_attr.shape[1],
            n_layers=config['n_layers'],
            hidden_channels=config['n_hidden'],
            latent_channels=config['n_latent'],
            n_unshared_layers=config['n_unshared_layers'],
            n_out=data.y.shape[1],
            aggr=(["sum", "max", "mean"] if config['aggr'] == "multi" else config['aggr'])
        ).to(device)
        
        optimizer = configure_optimizer(model, config['lr'], config['weight_decay'])
        
        # Set up logging
        log_file = tracker.logs_dir / f"env_gnn_fold_{k}.log"
        logger = TrainingLogger(log_file, f"Environment GNN Fold {k}")
        logger.write_header()
        
        # Training loop
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        
        for epoch in range(config['n_epochs']):
            # Adjust learning rate
            if epoch == int(config['n_epochs'] * 0.25):
                optimizer = configure_optimizer(model, config['lr']/5, config['weight_decay'])
            elif epoch == int(config['n_epochs'] * 0.5):
                optimizer = configure_optimizer(model, config['lr']/25, config['weight_decay'])
            elif epoch == int(config['n_epochs'] * 0.75):
                optimizer = configure_optimizer(model, config['lr']/125, config['weight_decay'])
            
            # Training
            train_loss = train_epoch_geometric(
                train_loader, model, optimizer, device,
                augment=True,
                augment_strength=config['augment_strength'],
                augment_edges=config['augment_edges']
            )
            
            # Validation
            valid_loss, predictions, targets, valid_idxs, is_central = validate_geometric(
                valid_loader, model, device, return_ids=True, return_centrals=True
            )
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            # Log progress
            logger.log_epoch(epoch, train_loss, valid_loss, predictions, targets)
            
            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_predictions = predictions.copy()
                best_targets = targets.copy()
                best_valid_idxs = valid_idxs.copy()
                best_is_central = is_central.copy()
        
        logger.close()
        
        # Record training results
        final_metrics = {
            'best_epoch': best_epoch,
            'best_valid_loss': best_loss,
            'final_train_loss': train_losses[-1],
            'final_valid_loss': valid_losses[-1]
        }
        
        tracker.record_model_training('env_gnn', k, config, final_metrics, model)
        
        # Prepare additional data for predictions
        def select_valid(data_tensor):
            return data_tensor[valid_indices].numpy().flatten()
        
        additional_data = {
            'subhalo_id': select_valid(data.subhalo_id),
            'log_Mhalo_dmo': select_valid(data.x[:, 0]),
            'log_Vmax_dmo': select_valid(data.x[:, 1]),
            'x_dmo': select_valid(data.pos[:, 0]),
            'y_dmo': select_valid(data.pos[:, 1]),
            'z_dmo': select_valid(data.pos[:, 2]),
            'is_central': select_valid(data.is_central).astype(int),
            'overdensity': select_valid(data.overdensity),
        }
        
        # Save predictions with detailed metadata
        pred_file = tracker.save_predictions(
            'env_gnn', k, best_predictions, best_targets, best_valid_idxs, additional_data
        )
        
        results[f'fold_{k}'] = {
            'final_metrics': final_metrics,
            'predictions_file': pred_file,
            'model_file': tracker.models_dir / f"env_gnn_fold_{k}.pth"
        }
        
        print(f"Fold {k} complete - Best validation loss: {best_loss:.5f}")
    
    return results


def train_merger_gnn_tracked(experiment_name: str, fold: int = None, 
                           residual_mode: bool = False, base_model: str = None) -> Dict[str, Any]:
    """Train Merger Tree GNN with full tracking.
    
    Args:
        experiment_name: Name of the experiment for tracking
        fold: Specific fold to train (if None, trains all folds)
        residual_mode: Whether to train on residuals from another model
        base_model: Base model name for residual learning
        
    Returns:
        Dictionary with training results and file paths
    """
    # Set up experiment tracking
    tracker = ExperimentTracker(experiment_name)
    config = get_model_config('merger_gnn')
    device = get_device()
    
    # Load merger trees
    with open(RESULTS_DIR / "merger_trees.pkl", "rb") as f:
        trees = pickle.load(f)
    
    # Create or load consistent splits on original trees
    splits = tracker.create_consistent_splits("merger_trees", "random")
    
    # Filter trees based on config and track valid indices
    if config['only_centrals']:
        valid_mask = [tree.is_central and tree.y[0, 0] > config['minimum_root_stellar_mass'] for tree in trees]
    else:
        valid_mask = [tree.y[0, 0] > config['minimum_root_stellar_mass'] for tree in trees]
    
    trees = [tree for i, tree in enumerate(trees) if valid_mask[i]]
    valid_original_indices = [i for i, mask in enumerate(valid_mask) if mask]
    
    # Create mapping from original indices to filtered indices
    index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_original_indices)}
    
    # Handle residual mode
    if residual_mode:
        if base_model is None:
            raise ValueError("base_model must be specified for residual_mode")
        # Load residual targets from the base experiment
        # The residual data is saved in the current experiment's predictions directory
        residual_file = tracker.predictions_dir / f"residuals_{base_model}.csv"
        
        if not residual_file.exists():
            raise FileNotFoundError(f"Residual data not found: {residual_file}")
        
        # Load residual data
        import pandas as pd
        residual_data = pd.read_csv(residual_file)
    
    # Determine which folds to train
    folds_to_train = [fold] if fold is not None else list(range(K_FOLDS))
    
    results = {}
    
    for k in folds_to_train:
        print(f"\n=== Training Merger Tree GNN - Fold {k+1}/{K_FOLDS} ===")
        
        # Get train/validation indices for trees and map to filtered indices
        original_train_indices = splits[k]['train']
        original_valid_indices = splits[k]['valid']
        
        # Map to filtered tree indices, excluding invalid trees
        train_indices = [index_mapping[i] for i in original_train_indices if i in index_mapping]
        valid_indices = [index_mapping[i] for i in original_valid_indices if i in index_mapping]
        
        train_trees = [trees[i] for i in train_indices]
        valid_trees = [trees[i] for i in valid_indices]
        
        # Create data loaders
        train_loader = GeometricDataLoader(train_trees, batch_size=config['batch_size'], shuffle=True)
        valid_loader = GeometricDataLoader(valid_trees, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = MultiSAGENet(
            n_in=8,  # Based on merger tree features
            n_hidden=config['n_hidden'],
            n_layers=config['n_layers'],
            bias=config['bias'],
            n_out=2
        ).to(device)
        
        optimizer = configure_optimizer(model, config['lr'], config['weight_decay'])
        
        # Determine model name for logging (before training loop)
        model_name = f'merger_gnn_residual_{base_model}' if residual_mode else 'merger_gnn'
        
        # Set up logging
        log_file = tracker.logs_dir / f"{model_name}_fold_{k}.log"
        logger_title = f"Merger Tree GNN (Residual {base_model}) Fold {k}" if residual_mode else f"Merger Tree GNN Fold {k}"
        logger = TrainingLogger(log_file, logger_title)
        logger.write_header()
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(config['n_epochs']):
            # Training
            train_loss = train_epoch_geometric(
                train_loader, model, optimizer, device,
                augment=True,
                augment_strength=config['augment_strength'],
                augment_edges=config['augment_edges']
            )
            
            # Validation
            valid_loss, predictions, targets, root_subhalo_ids = validate_geometric(
                valid_loader, model, device, return_ids=True, return_centrals=False
            )
            
            # Log progress
            logger.log_epoch(epoch, train_loss, valid_loss, predictions, targets)
            
            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_predictions = predictions.copy()
                best_targets = targets.copy()
                best_ids = root_subhalo_ids.copy()
        
        logger.close()
        
        # Record training results
        final_metrics = {
            'best_epoch': best_epoch,
            'best_valid_loss': best_loss
        }
        
        # Model name was already determined above for logging
        
        tracker.record_model_training(model_name, k, config, final_metrics, model)
        
        # Prepare additional data including is_central information and subhalo_ids
        val_indices = np.arange(len(best_predictions))  # Sequential validation indices
        additional_data = {
            'is_central': [valid_trees[i].is_central for i in range(len(valid_trees))],
            'subhalo_id': best_ids,  # Store actual subhalo IDs in additional_data
        }
        
        # Save predictions
        pred_file = tracker.save_predictions(
            model_name, k, best_predictions, best_targets, val_indices, additional_data
        )
        
        results[f'fold_{k}'] = {
            'final_metrics': final_metrics,
            'predictions_file': pred_file,
            'model_file': tracker.models_dir / f"{model_name}_fold_{k}.pth"
        }
        
        print(f"Fold {k} complete - Best validation loss: {best_loss:.5f}")
    
    return results


def train_mlp_tracked(experiment_name: str, fold: int = None) -> Dict[str, Any]:
    """Train MLP baseline with full tracking.
    
    Args:
        experiment_name: Name of the experiment for tracking
        fold: Specific fold to train (if None, trains all folds)
        
    Returns:
        Dictionary with training results and file paths
    """
    # Set up experiment tracking
    tracker = ExperimentTracker(experiment_name)
    config = get_model_config('mlp')
    device = get_device()
    
    # Load data (reuse cosmic graph data but only node features)
    with open(RESULTS_DIR / "cosmic_graphs.pkl", 'rb') as f:
        data = pickle.load(f)
    
    X = data.x
    y = data.y
    
    # Use the same splits as environment GNN for fair comparison
    splits = tracker.create_consistent_splits("cosmic_graph", "spatial")
    
    # Determine which folds to train
    folds_to_train = [fold] if fold is not None else list(range(K_FOLDS))
    
    results = {}
    
    for k in folds_to_train:
        print(f"\n=== Training MLP Baseline - Fold {k+1}/{K_FOLDS} ===")
        
        # Get train/validation indices
        train_indices = splits[k]['train']
        valid_indices = splits[k]['valid']
        
        # Create datasets and loaders
        from .utils import SubhaloDataset
        train_dataset = SubhaloDataset(X[train_indices], y[train_indices])
        valid_dataset = SubhaloDataset(X[valid_indices], y[valid_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = nn.Sequential(
            nn.Linear(X.shape[1], config['n_hidden']),
            nn.ReLU(),
            nn.Linear(config['n_hidden'], config['n_hidden']),
            nn.ReLU(),
            nn.Linear(config['n_hidden'], config['n_hidden']),
            nn.ReLU(),
            nn.Linear(config['n_hidden'], 2 * y.shape[1])  # *2 for mean + logvar
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Set up logging
        log_file = tracker.logs_dir / f"mlp_fold_{k}.log"
        logger = TrainingLogger(log_file, f"MLP Baseline Fold {k}")
        logger.write_header()
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(config['n_epochs']):
            # Training
            train_loss = train_epoch_tensor(train_loader, model, optimizer, device)
            
            # Validation
            valid_loss, predictions, targets = validate_tensor(valid_loader, model, device)
            
            # Log progress
            logger.log_epoch(epoch, train_loss, valid_loss, predictions, targets)
            
            # Save best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_predictions = predictions.copy()
                best_targets = targets.copy()
        
        logger.close()
        
        # Record training results
        final_metrics = {
            'best_epoch': best_epoch,
            'best_valid_loss': best_loss
        }
        
        tracker.record_model_training('mlp', k, config, final_metrics, model)
        
        # Get subhalo IDs for the validation set
        subhalo_ids = data.subhalo_id[valid_indices].numpy()
        
        # Prepare additional data for MLP predictions
        additional_data = {
            'is_central': data.is_central[valid_indices].numpy().astype(int),
        }
        
        # Save predictions with validation indices and subhalo_ids in additional_data
        val_indices = np.arange(len(best_predictions))  # Sequential validation indices
        additional_data['subhalo_id'] = subhalo_ids
        
        pred_file = tracker.save_predictions(
            'mlp', k, best_predictions, best_targets, val_indices, additional_data
        )
        
        results[f'fold_{k}'] = {
            'final_metrics': final_metrics,
            'predictions_file': pred_file,
            'model_file': tracker.models_dir / f"mlp_fold_{k}.pth"
        }
        
        print(f"Fold {k} complete - Best validation loss: {best_loss:.5f}")
    
    return results


def run_comparison_experiment(experiment_name: str, models: List[str] = None) -> ExperimentTracker:
    """Run a complete comparison experiment with all models.
    
    Args:
        experiment_name: Name of the comparison experiment
        models: List of models to train ['mlp', 'env_gnn', 'merger_gnn']
        
    Returns:
        ExperimentTracker with all results
    """
    if models is None:
        models = ['mlp', 'env_gnn', 'merger_gnn']
    
    print(f"Starting comparison experiment: {experiment_name}")
    print(f"Models to train: {models}")
    
    # Train each model
    if 'mlp' in models:
        print("\n" + "="*50)
        print("TRAINING MLP BASELINE")
        print("="*50)
        mlp_results = train_mlp_tracked(experiment_name)
    
    if 'env_gnn' in models:
        print("\n" + "="*50)
        print("TRAINING ENVIRONMENT GNN")
        print("="*50)
        env_results = train_env_gnn_tracked(experiment_name)
    
    if 'merger_gnn' in models:
        print("\n" + "="*50)
        print("TRAINING MERGER TREE GNN")
        print("="*50)
        merger_results = train_merger_gnn_tracked(experiment_name)
    
    # Create tracker and evaluate
    tracker = ExperimentTracker(experiment_name)
    
    print("\n" + "="*50)
    print("EVALUATING ALL MODELS")
    print("="*50)
    
    # Evaluate all models
    eval_results = tracker.evaluate_models(['mse', 'mae', 'r2', 'pearson'])
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    summary = tracker.get_experiment_summary()
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Models trained: {len(summary['metadata']['models'])}")
    print(f"Prediction files: {summary['files']['predictions']}")
    
    if 'best_models' in summary:
        print("\nBest models by metric:")
        for metric, model in summary['best_models'].items():
            print(f"  {metric.upper()}: {model}")
    
    print(f"\nResults saved in: {tracker.exp_dir}")
    
    return tracker


def run_residual_experiment(base_experiment: str, base_model: str, 
                          residual_models: List[str] = None) -> ExperimentTracker:
    """Run residual learning experiment in the same directory as base experiment.
    
    Args:
        base_experiment: Name of experiment with base model predictions
        base_model: Base model to compute residuals from
        residual_models: Models to train on residuals
        
    Returns:
        ExperimentTracker for base experiment (residual results stored alongside)
    """
    if residual_models is None:
        residual_models = ['merger_gnn']
    
    print(f"Starting residual experiment in: {base_experiment}")
    print(f"Base model: {base_model}")
    print(f"Residual models: {residual_models}")
    
    # Set up residual experiment (uses same tracker as base experiment)
    from experiment_tracker import run_residual_experiment as setup_residual
    tracker = setup_residual(base_experiment, base_model, f"{base_experiment}_residuals_{base_model}")
    
    # Train residual models
    if 'merger_gnn' in residual_models:
        merger_results = train_merger_gnn_tracked(
            base_experiment, 
            residual_mode=True, 
            base_model=base_model
        )
    
    return tracker