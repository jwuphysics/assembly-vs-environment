# Assembly versus Environment

Is halo assembly history or environment more important for learning galaxy properties? 

This repository implements a comprehensive comparison framework using graph neural networks to predict galaxy properties from either environmental context or assembly history. The framework supports **multi-output predictions**, simultaneously predicting both stellar mass and gas mass from TNG data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up TNG data paths (optional - defaults to ./illustris_data/)
export TNG_BASE_PATH="/path/to/your/TNG100-1"
export TNG_EXTENDED_TREE_PATH="/path/to/extended/tree/files"

# Run a complete comparison experiment
python run_experiments.py --experiment "my_comparison" --models mlp env_gnn merger_gnn

# Analyze results
python analysis_template.py --experiment "my_comparison"
```

## Experiment Framework

### Reproducible Comparisons
The experiment tracking system ensures:
- **Consistent train/validation splits** across all models
- **Detailed prediction recording** for example-by-example analysis
- **Automatic artifact management** (models, logs, predictions)
- **Cross-model evaluation** with standardized metrics

### Supported Models
1. **MLP Baseline**: Simple multilayer perceptron using only halo properties
2. **Environment GNN**: Graph neural network using 3D environmental context  
3. **Merger Tree GNN**: Graph neural network using assembly history

All models support **multi-output predictions**, simultaneously predicting:
- **Stellar mass** (log M_star) 
- **Gas mass** (log M_gas)
- **Uncertainty estimates** (log-variance) for both outputs

### Residual Learning
Train models to predict residuals from existing base model predictions:
```bash
# First train base model
python run_experiments.py --experiment "comparison" --models env_gnn

# Then train on residuals using existing predictions
python run_experiments.py --experiment "comparison" --residual-only --base-model env_gnn
```

## Data

**Environment** We use `FoF_subfind` subhalo catalogs from Illustris TNG100, crossmatched between dark matter only and hydrodynamic simulations. We select halos with $M_{\rm halo} \geq 10^{10} M_{\odot}$ and without any subhalo flags.

**Assembly history** We use the `Subfind` merger trees from TNG100-1-Dark, selecting only halos with $M_{\rm halo} \geq 10^{10} M_{\odot}$.

**Targets** Models predict multiple galaxy properties:
- `subhalo_logstellarmass`: Log stellar mass in solar masses
- `subhalo_loggasmass`: Log gas mass in solar masses

Both targets are extracted from hydrodynamic simulations and predicted using dark matter-only features.

## Code Structure

```
src/
├── config.py              # Centralized configuration
├── data.py                 # TNG data loading utilities  
├── loader.py               # Graph construction and data preparation
├── model.py                # Neural network architectures
├── training_utils.py       # Shared training/validation functions
├── experiment_tracker.py   # Experiment management and tracking
└── train_with_tracking.py  # High-level training workflows

run_experiments.py          # Main experiment runner
analysis_template.py        # Result analysis and visualization
requirements.txt            # Python dependencies
```

## Configuration

All hyperparameters are centralized in `src/config.py`:

```python
from src.config import get_model_config

# Get configuration for specific model
env_config = get_model_config('env_gnn')
merger_config = get_model_config('merger_gnn')
mlp_config = get_model_config('mlp')
```

## Environment Variables

- `TNG_BASE_PATH`: Path to TNG100-1 data directory
- `TNG_EXTENDED_TREE_PATH`: Path to extended merger tree files

## Example Workflows

### Basic Comparison
```python
from src.train_with_tracking import run_comparison_experiment

# Train all models with consistent splits
tracker = run_comparison_experiment("my_experiment", ["mlp", "env_gnn", "merger_gnn"])

# Get combined predictions for analysis
combined = tracker.combine_all_predictions()
eval_results = tracker.evaluate_models()
```

### Residual Learning
```python
from src.train_with_tracking import run_residual_experiment

# Train merger tree GNN on environment GNN residuals
residual_tracker = run_residual_experiment(
    base_experiment="my_experiment",
    base_model="env_gnn", 
    residual_models=["merger_gnn"]
)
```

### Custom Analysis
```python
from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("my_experiment")

# Load specific model predictions
env_preds = tracker.load_predictions("env_gnn", fold=0)
merger_preds = tracker.load_predictions("merger_gnn", fold=0)

# Create residual targets
residuals = tracker.create_residual_targets("env_gnn")
```

## Key Features

- **Multi-Output Predictions**: Simultaneously predict stellar mass and gas mass with uncertainty estimates
- **Consistent Splits**: Same train/validation splits across all models
- **Detailed Tracking**: Every prediction saved with metadata for analysis
- **Configurable**: Easy to modify hyperparameters and add new models
- **Portable**: No hardcoded paths, works on any system
- **Reproducible**: Fixed random seeds and comprehensive logging

## Output Format

Results are saved with separate columns for each target:
- `pred_{model}_Mstar`: Stellar mass predictions
- `pred_{model}_Mgas`: Gas mass predictions  
- `target_Mstar`: True stellar mass values
- `target_Mgas`: True gas mass values

## Command Line Options

The main experiment runner supports these options:

- `--experiment NAME`: Required experiment name
- `--models [mlp,env_gnn,merger_gnn]`: Which models to train (default: all)
- `--residual`: Run residual experiment after main experiment  
- `--residual-only`: Only run residual experiment using existing base model
- `--base-model MODEL`: Base model for residual learning (default: env_gnn)
- `--evaluate-only`: Only evaluate existing experiment results

### Examples

```bash
# Train all models
python run_experiments.py --experiment "my_test" --models mlp env_gnn merger_gnn

# Evaluate existing experiment
python run_experiments.py --experiment "my_test" --evaluate-only

# Train residual model on existing base predictions  
python run_experiments.py --experiment "my_test" --residual-only --base-model env_gnn
```
