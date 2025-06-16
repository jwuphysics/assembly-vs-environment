"""
Experiment tracking and artifact management for assembly vs environment research.

This module provides utilities for:
1. Consistent train/validation splits across different models
2. Recording training artifacts and predictions
3. Enabling example-by-example evaluation
4. Supporting residual learning experiments
"""

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

# Import torch only when needed for model saving
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .config import RESULTS_DIR, LOGS_DIR, RANDOM_SEED, K_FOLDS
except ImportError:
    from config import RESULTS_DIR, LOGS_DIR, RANDOM_SEED, K_FOLDS


class ExperimentTracker:
    """Track experiments with consistent splits and detailed artifacts."""
    
    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        """Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing artifacts (defaults to RESULTS_DIR)
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir or RESULTS_DIR
        self.exp_dir = self.base_dir / f"experiment_{experiment_name}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment metadata
        self.metadata_file = self.exp_dir / "metadata.json"
        self.splits_file = self.exp_dir / "data_splits.pkl"
        self.predictions_dir = self.exp_dir / "predictions"
        self.models_dir = self.exp_dir / "models"
        self.logs_dir = self.exp_dir / "logs"
        
        # Create subdirectories
        for dir_path in [self.predictions_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Load or initialize metadata
        self._load_or_create_metadata()
        
    def _load_or_create_metadata(self):
        """Load existing metadata or create new experiment."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "experiment_name": self.experiment_name,
                "created_at": datetime.now().isoformat(),
                "random_seed": RANDOM_SEED,
                "k_folds": K_FOLDS,
                "models": {},
                "data_hash": None,
                "splits_created": False
            }
            self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def create_consistent_splits(self, data_source: Union[str, Any], 
                                split_method: str = "spatial",
                                force_recreate: bool = False) -> Dict[int, Dict[str, np.ndarray]]:
        """Create consistent train/validation splits across all models.
        
        Args:
            data_source: Either 'cosmic_graph', 'merger_trees', or actual data object
            split_method: 'spatial' for cosmic graphs, 'random' for merger trees
            force_recreate: Whether to recreate splits even if they exist
            
        Returns:
            Dictionary mapping fold indices to {'train': indices, 'valid': indices}
        """
        # Check if splits already exist and are valid
        if self.splits_file.exists() and not force_recreate:
            with open(self.splits_file, 'rb') as f:
                existing_splits = pickle.load(f)
            if existing_splits.get('data_source') == data_source:
                print(f"Using existing splits for {data_source}")
                return existing_splits['splits']
        
        # Create new splits
        print(f"Creating new {split_method} splits for {data_source}")
        
        if data_source == "cosmic_graph":
            splits = self._create_spatial_splits()
        elif data_source == "merger_trees":
            splits = self._create_random_splits_trees()
        else:
            # Custom data object - determine split method
            if hasattr(data_source, 'pos'):  # Spatial data
                splits = self._create_spatial_splits(data_source)
            else:  # Non-spatial data
                splits = self._create_random_splits(data_source)
        
        # Save splits with metadata
        split_data = {
            'data_source': data_source,
            'split_method': split_method,
            'splits': splits,
            'created_at': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'k_folds': K_FOLDS
        }
        
        with open(self.splits_file, 'wb') as f:
            pickle.dump(split_data, f)
        
        # Update metadata
        self.metadata['splits_created'] = True
        self.metadata['data_source'] = str(data_source)
        self.metadata['split_method'] = split_method
        self._save_metadata()
        
        return splits
    
    def _create_spatial_splits(self, data=None) -> Dict[int, Dict[str, np.ndarray]]:
        """Create spatial splits for cosmic graph data."""
        if data is None:
            # Load cosmic graph data
            with open(RESULTS_DIR / "cosmic_graphs.pkl", 'rb') as f:
                data = pickle.load(f)
        
        try:
            from .utils import get_spatial_train_valid_indices
        except ImportError:
            from utils import get_spatial_train_valid_indices
        
        splits = {}
        for k in range(K_FOLDS):
            train_indices, valid_indices = get_spatial_train_valid_indices(data, k=k, K=K_FOLDS)
            splits[k] = {
                'train': train_indices.numpy() if hasattr(train_indices, 'numpy') else train_indices,
                'valid': valid_indices.numpy() if hasattr(valid_indices, 'numpy') else valid_indices
            }
        
        return splits
    
    def _create_random_splits_trees(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Create random splits for merger tree data."""
        # Load merger trees
        with open(RESULTS_DIR / "merger_trees.pkl", "rb") as f:
            trees = pickle.load(f)
        
        N = len(trees)
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(N)
        
        splits = {}
        for k in range(K_FOLDS):
            valid_start = int(k / K_FOLDS * N)
            valid_end = int((k + 1) / K_FOLDS * N)
            
            valid_indices = indices[valid_start:valid_end]
            train_indices = np.concatenate([indices[:valid_start], indices[valid_end:]])
            
            splits[k] = {
                'train': train_indices,
                'valid': valid_indices
            }
        
        return splits
    
    def _create_random_splits(self, data) -> Dict[int, Dict[str, np.ndarray]]:
        """Create random splits for generic data."""
        N = len(data)
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(N)
        
        splits = {}
        for k in range(K_FOLDS):
            valid_start = int(k / K_FOLDS * N)
            valid_end = int((k + 1) / K_FOLDS * N)
            
            valid_indices = indices[valid_start:valid_end]
            train_indices = np.concatenate([indices[:valid_start], indices[valid_end:]])
            
            splits[k] = {
                'train': train_indices,
                'valid': valid_indices
            }
        
        return splits
    
    def record_model_training(self, model_name: str, fold: int, 
                            model_config: Dict, final_metrics: Dict,
                            model_state: Optional[Any] = None):
        """Record model training information.
        
        Args:
            model_name: Name of the model (e.g., 'env_gnn', 'merger_gnn', 'mlp')
            fold: Cross-validation fold index
            model_config: Configuration used for training
            final_metrics: Final training/validation metrics
            model_state: Trained model state dict (optional)
        """
        # Initialize model entry if needed
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {}
        
        # Record fold information
        fold_info = {
            'config': model_config,
            'final_metrics': final_metrics,
            'trained_at': datetime.now().isoformat(),
            'model_saved': False
        }
        
        # Save model if provided
        if model_state is not None:
            if not TORCH_AVAILABLE:
                print(f"Warning: torch not available, cannot save model {model_name}_fold_{fold}")
                fold_info['model_saved'] = False
            else:
                model_file = self.models_dir / f"{model_name}_fold_{fold}.pth"
                torch.save(model_state.state_dict(), model_file)
                fold_info['model_saved'] = True
                fold_info['model_file'] = str(model_file)
        
        self.metadata['models'][model_name][f'fold_{fold}'] = fold_info
        self._save_metadata()
    
    def save_predictions(self, model_name: str, fold: int, 
                        predictions: np.ndarray, targets: np.ndarray,
                        indices: np.ndarray, additional_data: Optional[Dict] = None) -> Path:
        """Save model predictions with detailed metadata.
        
        Args:
            model_name: Name of the model
            fold: Cross-validation fold index
            predictions: Model predictions
            targets: Ground truth targets
            indices: Data indices (for matching across models)
            additional_data: Additional data columns (e.g., features, metadata)
            
        Returns:
            Path to saved predictions file
        """
        # Create comprehensive prediction record
        pred_data = {
            'fold': fold,
            'indices': indices,
            'predictions': predictions,
            'targets': targets,
            'model_name': model_name,
            'experiment': self.experiment_name,
            'created_at': datetime.now().isoformat()
        }
        
        # Add additional data if provided
        if additional_data:
            pred_data.update(additional_data)
        
        # Convert to DataFrame for easy analysis
        df_data = {
            'fold': fold,
            'index': indices,
        }
        
        # Handle multi-output predictions and targets
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-output case
            for i in range(predictions.shape[1]):
                output_name = 'Mstar' if i == 0 else 'Mgas' if i == 1 else f'output_{i}'
                df_data[f'pred_{model_name}_{output_name}'] = predictions[:, i]
                df_data[f'target_{output_name}'] = targets[:, i]
        else:
            # Single output case
            df_data[f'pred_{model_name}'] = predictions.flatten() if predictions.ndim > 1 else predictions
            df_data['target'] = targets.flatten() if targets.ndim > 1 else targets
        
        # Add additional columns
        if additional_data:
            for key, values in additional_data.items():
                if isinstance(values, (np.ndarray, list)) and len(values) == len(indices):
                    df_data[key] = values
        
        df = pd.DataFrame(df_data)
        
        # Save both formats
        pred_file_pkl = self.predictions_dir / f"{model_name}_fold_{fold}_predictions.pkl"
        pred_file_csv = self.predictions_dir / f"{model_name}_fold_{fold}_predictions.csv"
        
        with open(pred_file_pkl, 'wb') as f:
            pickle.dump(pred_data, f)
        
        df.to_csv(pred_file_csv, index=False)
        
        print(f"Saved predictions to {pred_file_csv}")
        return pred_file_csv
    
    def load_predictions(self, model_name: str, fold: int) -> pd.DataFrame:
        """Load predictions for a specific model and fold."""
        pred_file = self.predictions_dir / f"{model_name}_fold_{fold}_predictions.csv"
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions not found: {pred_file}")
        return pd.read_csv(pred_file)
    
    def combine_all_predictions(self) -> pd.DataFrame:
        """Combine predictions from all models and folds into a single DataFrame."""
        all_predictions = []
        
        # Find all prediction files
        pred_files = list(self.predictions_dir.glob("*_predictions.csv"))
        
        for pred_file in pred_files:
            df = pd.read_csv(pred_file)
            # Extract model name and fold from filename
            filename_parts = pred_file.stem.split('_')
            model_name = '_'.join(filename_parts[:-3])  # Everything before _fold_X_predictions
            fold = int(filename_parts[-2])
            
            # Rename prediction column to be model-specific
            pred_col = f'pred_{model_name}'
            if pred_col in df.columns:
                df = df.rename(columns={pred_col: f'pred_{model_name}'})
            
            all_predictions.append(df)
        
        if not all_predictions:
            raise ValueError("No prediction files found")
        
        # Merge all predictions on index and fold
        combined = all_predictions[0]
        
        # Determine merge keys based on available columns
        merge_keys = ['index', 'fold']
        # Add target columns that exist in all dataframes
        target_cols = [col for col in combined.columns if col.startswith('target_')]
        if not target_cols and 'target' in combined.columns:
            target_cols = ['target']
        
        for target_col in target_cols:
            if all(target_col in df.columns for df in all_predictions):
                merge_keys.append(target_col)
        
        for df in all_predictions[1:]:
            # Merge on common keys, keeping all prediction columns
            combined = combined.merge(df, on=merge_keys, how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            combined = combined.loc[:, ~combined.columns.str.endswith('_dup')]
        
        return combined
    
    def create_residual_targets(self, base_model: str) -> pd.DataFrame:
        """Create residual targets for training models on prediction errors.
        
        Args:
            base_model: Name of the base model (e.g., 'env_gnn')
            
        Returns:
            DataFrame with residuals as new targets for both Mstar and Mgas
        """
        combined = self.combine_all_predictions()
        
        # Check for multi-output predictions
        base_pred_mstar = f'pred_{base_model}_Mstar'
        base_pred_mgas = f'pred_{base_model}_Mgas'
        
        if base_pred_mstar not in combined.columns or base_pred_mgas not in combined.columns:
            raise ValueError(f"Base model predictions not found: {base_pred_mstar}, {base_pred_mgas}")
        
        # Calculate residuals for both targets
        combined[f'residual_{base_model}_Mstar'] = combined['target_Mstar'] - combined[base_pred_mstar]
        combined[f'residual_{base_model}_Mgas'] = combined['target_Mgas'] - combined[base_pred_mgas]
        
        # Save residual targets
        residual_file = self.predictions_dir / f"residuals_{base_model}.csv"
        combined.to_csv(residual_file, index=False)
        
        print(f"Created residual targets: {residual_file}")
        return combined
    
    def evaluate_models(self, metrics: List[str] = None, min_stellar_mass: Optional[float] = None, 
                       only_centrals: bool = False) -> pd.DataFrame:
        """Evaluate all models with detailed metrics.
        
        Args:
            metrics: List of metrics to compute ['rmse', 'mae', 'r2', 'pearson']
            min_stellar_mass: Minimum stellar mass cut (log10 scale) for evaluation.
                            Only applied to stellar mass (Mstar) evaluations, leaving
                            gas mass (Mgas) evaluations unchanged.
            only_centrals: If True, only evaluate central galaxies. This ensures
                         fair comparison between models when merger tree GNN was
                         trained only on centrals (only_centrals=True).
            
        Returns:
            DataFrame with evaluation results
            
        Examples:
            >>> tracker = ExperimentTracker("my_experiment")
            >>> 
            >>> # Evaluate all galaxies (default)
            >>> results_all = tracker.evaluate_models()
            >>> 
            >>> # Fair comparison: centrals only (matches merger tree training)
            >>> results_centrals = tracker.evaluate_models(only_centrals=True)
            >>> 
            >>> # High-mass centrals only
            >>> results_high_mass_centrals = tracker.evaluate_models(
            ...     min_stellar_mass=10.0, only_centrals=True)
            >>> 
            >>> # Compare performance across mass ranges
            >>> low_mass = tracker.evaluate_models(min_stellar_mass=8.5)
            >>> high_mass = tracker.evaluate_models(min_stellar_mass=10.5)
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'pearson']
        
        # Find all prediction files and evaluate each separately
        pred_files = list(self.predictions_dir.glob("*_predictions.csv"))
        
        # Also include residual prediction files (different naming pattern)
        residual_pred_files = list(self.predictions_dir.glob("*residual*.csv"))
        # Filter out the residual target files (which don't contain predictions)
        residual_pred_files = [f for f in residual_pred_files if not f.stem.startswith('residuals_')]
        pred_files.extend(residual_pred_files)
        
        results = []
        for pred_file in pred_files:
            # Extract model name and fold from filename
            filename_parts = pred_file.stem.split('_')
            
            if '_predictions' in pred_file.stem:
                # Standard naming: model_fold_X_predictions.csv
                model_name = '_'.join(filename_parts[:-3])  # Everything before _fold_X_predictions
                fold = int(filename_parts[-2])
            elif 'residual' in pred_file.stem:
                # Residual naming: predictions_basemodel_residuals-residualmodel-fold.csv
                if '-' in pred_file.stem:
                    # Extract fold from end (e.g., -3of3 or -1of3)
                    fold_part = pred_file.stem.split('-')[-1]
                    if 'of' in fold_part:
                        fold = int(fold_part.split('of')[0]) - 1  # Convert to 0-based indexing
                    else:
                        fold = 0  # Default fold
                    # Extract residual model name
                    model_name = pred_file.stem.split('-')[1] + '_residual'
                else:
                    fold = 0
                    model_name = 'residual_model'
            else:
                # Fallback
                model_name = pred_file.stem
                fold = 0
            
            # Load individual prediction file
            df = pd.read_csv(pred_file)
            
            # Find prediction columns in this file
            pred_cols = [col for col in df.columns if col.startswith('pred_')]
            
            for pred_col in pred_cols:
                # Extract target type from prediction column
                # e.g., 'pred_env_gnn_Mstar' -> target_type='Mstar'
                parts = pred_col.replace('pred_', '').split('_')
                if len(parts) > 1 and parts[-1] in ['Mstar', 'Mgas']:
                    target_type = parts[-1]
                    target_col = f'target_{target_type}'
                else:
                    target_col = 'target'
                    target_type = 'default'
                
                if target_col not in df.columns:
                    print(f"Warning: target column {target_col} not found for {pred_col}")
                    continue
                
                predictions = df[pred_col].values
                targets = df[target_col].values
                
                # Remove NaN values and missing data (target < 0)
                valid_mask = ~(np.isnan(predictions) | np.isnan(targets) | (targets < 0))
                
                # Apply stellar mass cut if specified and target is stellar mass
                if min_stellar_mass is not None and target_type == 'Mstar':
                    stellar_mass_mask = targets >= min_stellar_mass
                    valid_mask = valid_mask & stellar_mass_mask
                
                # Apply centrals-only filter if specified
                if only_centrals and 'is_central' in df.columns:
                    centrals_mask = df['is_central'].values.astype(bool)
                    valid_mask = valid_mask & centrals_mask
                elif only_centrals:
                    print(f"Warning: only_centrals=True but 'is_central' column not found in {pred_col}")
                
                predictions = predictions[valid_mask]
                targets = targets[valid_mask]
                
                if len(predictions) == 0:
                    continue
                
                # Calculate metrics
                fold_metrics = {
                    'model': model_name, 
                    'target_type': target_type,
                    'fold': fold, 
                    'n_samples': len(predictions)
                }
                
                if 'rmse' in metrics:
                    fold_metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
                if 'mae' in metrics:
                    fold_metrics['mae'] = np.mean(np.abs(predictions - targets))
                if 'r2' in metrics:
                    ss_res = np.sum((targets - predictions) ** 2)
                    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                    fold_metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-8))
                if 'pearson' in metrics:
                    correlation = np.corrcoef(predictions, targets)[0, 1]
                    fold_metrics['pearson'] = correlation
                
                results.append(fold_metrics)
        
        results_df = pd.DataFrame(results)
        
        # Save evaluation results
        eval_file = self.exp_dir / "evaluation_results.csv"
        results_df.to_csv(eval_file, index=False)
        
        print(f"Evaluation results saved: {eval_file}")
        return results_df
    
    def get_experiment_summary(self) -> Dict:
        """Get a summary of the experiment."""
        summary = {
            'experiment_name': self.experiment_name,
            'metadata': self.metadata,
            'files': {
                'predictions': len(list(self.predictions_dir.glob("*.csv"))),
                'models': len(list(self.models_dir.glob("*.pth"))),
                'logs': len(list(self.logs_dir.glob("*.log")))
            }
        }
        
        # Add evaluation summary if available
        eval_file = self.exp_dir / "evaluation_results.csv"
        if eval_file.exists():
            eval_df = pd.read_csv(eval_file)
            summary['best_models'] = {}
            for metric in ['rmse', 'mae', 'r2', 'pearson']:
                if metric in eval_df.columns:
                    if metric in ['rmse', 'mae']:  # Lower is better
                        best_idx = eval_df.groupby('model')[metric].mean().idxmin()
                    else:  # Higher is better
                        best_idx = eval_df.groupby('model')[metric].mean().idxmax()
                    summary['best_models'][metric] = best_idx
        
        return summary


# Convenience functions for common workflows

def setup_comparison_experiment(experiment_name: str, 
                              data_sources: List[str] = None) -> ExperimentTracker:
    """Set up an experiment for comparing multiple models.
    
    Args:
        experiment_name: Name of the comparison experiment
        data_sources: List of data sources to create splits for
        
    Returns:
        Configured ExperimentTracker
    """
    if data_sources is None:
        data_sources = ["cosmic_graph", "merger_trees"]
    
    tracker = ExperimentTracker(experiment_name)
    
    # Create consistent splits for all data sources
    for data_source in data_sources:
        if data_source == "cosmic_graph":
            splits = tracker.create_consistent_splits("cosmic_graph", "spatial")
        elif data_source == "merger_trees":
            splits = tracker.create_consistent_splits("merger_trees", "random")
    
    print(f"Experiment '{experiment_name}' set up with consistent splits")
    return tracker


def run_residual_experiment(base_experiment: str, base_model: str, 
                          residual_experiment: str) -> ExperimentTracker:
    """Set up a residual learning experiment in the same directory as base experiment.
    
    Args:
        base_experiment: Name of experiment with base model predictions
        base_model: Name of base model to compute residuals from
        residual_experiment: Name for residual experiment (ignored, kept for compatibility)
        
    Returns:
        Base ExperimentTracker (residual results stored alongside base results)
    """
    # Load base experiment
    base_tracker = ExperimentTracker(base_experiment)
    
    # Create residual targets if they don't exist
    residual_file = base_tracker.predictions_dir / f"residuals_{base_model}.csv"
    if not residual_file.exists():
        residual_data = base_tracker.create_residual_targets(base_model)
        print(f"Created residual targets for {base_model}")
    else:
        print(f"Using existing residual targets for {base_model}")
    
    print(f"Residual experiment set up in '{base_experiment}' directory")
    print(f"Residual model predictions will be saved with '_residual_{base_model}' suffix")
    return base_tracker