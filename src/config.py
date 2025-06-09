"""
Configuration file for assembly vs environment research.

This module centralizes hyperparameters and configuration settings used across
different training scripts to improve maintainability and reproducibility.
"""

import os
from pathlib import Path

# ===============================
# PATH CONFIGURATION
# ===============================

ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# TNG data paths - can be overridden via environment variables
TNG_BASE_PATH = os.environ.get('TNG_BASE_PATH', f"{ROOT}/illustris_data/TNG100-1")
TNG_EXTENDED_TREE_PATH = os.environ.get('TNG_EXTENDED_TREE_PATH', f"{TNG_BASE_PATH}/files/sublink")

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ===============================
# GLOBAL SETTINGS
# ===============================

# Random seed for reproducibility
RANDOM_SEED = 42

# Device preference
DEVICE_PREFERENCE = "cuda"  # "cuda" or "cpu"

# TNG simulation parameters
BOXSIZE = 75 / 0.6774  # box size in comoving Mpc/h
H = 0.6774  # reduced Hubble constant
SNAPSHOT = 99

# ===============================
# DATA PROCESSING
# ===============================

# Data cuts and filtering
DATA_CUTS = {
    "minimum_log_halo_mass": 10,
}

# Graph construction parameters
D_LINK = 3  # Mpc - linking length for environmental graphs

# ===============================
# TRAINING HYPERPARAMETERS
# ===============================

# Cross-validation
K_FOLDS = 3

# Environment GNN training
ENV_GNN_CONFIG = {
    "n_epochs": 500,
    "num_parts": 32,  # for ClusterLoader
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "n_layers": 2,
    "n_hidden": 32,
    "n_latent": 8,
    "n_unshared_layers": 4,
    "aggr": "multi",  # "multi" uses ["sum", "max", "mean"]
    "augment_strength": 3e-3,
    "augment_edges": True,
}

# Merger tree GNN training
MERGER_GNN_CONFIG = {
    "minimum_root_stellar_mass": 8.5,
    "only_centrals": True,
    "n_epochs": 500,
    "batch_size": 128,
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "n_layers": 8,
    "n_hidden": 8,
    "bias": False,
    "augment_strength": 3e-4,
    "augment_edges": False,
}

# MLP baseline training
MLP_CONFIG = {
    "n_epochs": 200,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "n_hidden": 128,
}

# ===============================
# MODEL ARCHITECTURES
# ===============================

# EdgeInteractionGNN default parameters
EDGE_GNN_DEFAULTS = {
    "hidden_channels": 64,
    "latent_channels": 64,
    "n_unshared_layers": 4,
    "aggr": ["sum", "max", "mean"],
}

# MultiSAGENet default parameters  
MULTI_SAGE_DEFAULTS = {
    "n_hidden": 16,
    "n_layers": 4,
    "bias": True,
    "aggr": ["max", "mean"],
}

# ===============================
# LOGGING CONFIGURATION
# ===============================

# Logging intervals and formats
LOG_CONFIG = {
    "epoch_interval": 10,  # Log every N epochs
    "time_format": "%.1f",  # Time precision
    "loss_format": "%.5f",  # Loss precision
    "rmse_format": "%.6f",  # RMSE precision
}

# ===============================
# UTILITY FUNCTIONS
# ===============================

def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type.
    
    Args:
        model_type: One of 'env_gnn', 'merger_gnn', 'mlp'
        
    Returns:
        Configuration dictionary for the specified model
    """
    configs = {
        'env_gnn': ENV_GNN_CONFIG,
        'merger_gnn': MERGER_GNN_CONFIG, 
        'mlp': MLP_CONFIG,
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(configs.keys())}")
    
    return configs[model_type].copy()


def get_data_paths() -> dict:
    """Get all configured data paths.
    
    Returns:
        Dictionary with all path configurations
    """
    return {
        'root': ROOT,
        'results': RESULTS_DIR,
        'logs': LOGS_DIR,
        'tng_base': TNG_BASE_PATH,
        'tng_extended_trees': TNG_EXTENDED_TREE_PATH,
    }


def validate_paths() -> bool:
    """Validate that required paths exist.
    
    Returns:
        True if all paths are valid, raises exception otherwise
    """
    required_paths = [TNG_BASE_PATH]
    missing_paths = [path for path in required_paths if not Path(path).exists()]
    
    if missing_paths:
        raise FileNotFoundError(
            f"Required data paths not found: {missing_paths}\n"
            f"Set appropriate environment variables or ensure data is downloaded."
        )
    
    return True


# ===============================
# ENVIRONMENT VALIDATION
# ===============================

def print_config_summary():
    """Print a summary of current configuration."""
    print("=== Assembly vs Environment Configuration ===")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Device preference: {DEVICE_PREFERENCE}")
    print(f"K-fold splits: {K_FOLDS}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"TNG base path: {TNG_BASE_PATH}")
    print(f"TNG extended trees: {TNG_EXTENDED_TREE_PATH}")
    print("=" * 45)


if __name__ == "__main__":
    print_config_summary()
    validate_paths()
    print("Configuration validation successful!")