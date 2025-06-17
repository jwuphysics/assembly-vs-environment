# CLAUDE.md - Assembly vs Environment Project Usage

## Scientific Goal
This project investigates whether **halo assembly history** or **environmental context** is more important for predicting galaxy properties (stellar mass and gas mass) using graph neural networks on IllustrisTNG data.

## Typical Experiment Workflow

### 1. Running Complete Comparisons
```bash
# Standard three-model comparison
python run_experiments.py --experiment "my_comparison" --models mlp env_gnn merger_gnn

# Environment vs merger tree only
python run_experiments.py --experiment "env_vs_merger" --models env_gnn merger_gnn

# With residual analysis
python run_experiments.py --experiment "with_residuals" --models mlp env_gnn merger_gnn --residual --base-model env_gnn
```

### 2. Analyzing Results
```bash
# Evaluate existing experiment
python run_experiments.py --experiment "my_comparison" --evaluate-only

# Evaluate with stellar mass cut (log₁₀(M_star) ≥ 10.0)
python run_experiments.py --experiment "my_comparison" --evaluate-only --min-stellar-mass 10.0

# Fair comparison: evaluate only central galaxies
python run_experiments.py --experiment "my_comparison" --evaluate-only --only-centrals

# High-mass centrals only for fair comparison
python run_experiments.py --experiment "my_comparison" --evaluate-only --min-stellar-mass 10.0 --only-centrals

# Full analysis with plots
python analysis_template.py --experiment "my_comparison"

# Analysis with filters for fair comparison
python analysis_template.py --experiment "my_comparison" --min-stellar-mass 10.0 --only-centrals
```

### 3. Data Requirements
- Set environment variables (optional):
  ```bash
  export TNG_BASE_PATH="/path/to/TNG100-1"
  export TNG_EXTENDED_TREE_PATH="/path/to/extended/trees"
  ```
- Data automatically loaded from `illustris_data/` if paths not set

## Key Scientific Features

### Models Compared
1. **MLP Baseline**: Simple multilayer perceptron using only halo mass and Vmax
2. **Environment GNN**: Graph neural network using 3D spatial neighborhood graphs
3. **Merger Tree GNN**: Graph neural network using assembly history trees

### Multi-Output Predictions
All models predict both:
- **Stellar mass** (log M_star) 
- **Gas mass** (log M_gas)
- **Uncertainties** (log-variance for both)

### Consistent Evaluation
- **Spatial cross-validation** for environment models (avoids spatial correlation)
- **Random cross-validation** for merger tree models
- **Identical train/validation splits** across all models for fair comparison
- **Example-by-example predictions** saved for detailed analysis

### Residual Learning
Test if merger trees can improve upon environmental predictions:
```python
# Train base environment model first
python run_experiments.py --experiment "base" --models env_gnn

# Train merger tree on environment residuals  
python run_experiments.py --experiment "base" --residual --base-model env_gnn
```

## Important Files and Outputs

### Results Structure
```
results/experiment_{name}/
├── metadata.json              # Experiment configuration and timing
├── data_splits.pkl           # Consistent train/validation splits
├── predictions/              # Model predictions with metadata
│   ├── {model}_fold_{k}_predictions.csv
│   └── residuals_{base_model}.csv
├── models/                   # Trained model weights
├── logs/                     # Training logs
└── evaluation_results.csv    # Cross-validation metrics
```

### Key Scripts
- `run_experiments.py`: Main experiment runner with tracking
- `analysis_template.py`: Comprehensive result analysis and visualization
- `src/train.py`: Training functions with full artifact management
- `src/experiment_tracker.py`: Experiment management and consistent splits

## Programmatic Usage

### Quick Analysis
```python
from src.experiment_tracker import ExperimentTracker

# Load experiment results
tracker = ExperimentTracker("my_comparison")

# Get comprehensive results table with ALL predictions matched by subhalo_id
# Includes: standard models, residual models, all metadata, matched across folds
combined = tracker.combine_all_predictions()
print(f"Combined: {len(combined)} galaxies, {len(combined.columns)} columns")

# Evaluate all models
results = tracker.evaluate_models(['mse', 'mae', 'r2', 'pearson'])

# Evaluate with stellar mass cut (only log₁₀(M_star) ≥ 10.0)
results_high_mass = tracker.evaluate_models(['mse', 'mae', 'r2', 'pearson'], min_stellar_mass=10.0)

# Fair comparison: centrals only (matches merger tree training)
results_fair = tracker.evaluate_models(['mse', 'mae', 'r2', 'pearson'], only_centrals=True)

# Get experiment summary
summary = tracker.get_experiment_summary()
```

### Custom Workflows
```python
from src.train_with_tracking import run_comparison_experiment

# Run full comparison
tracker = run_comparison_experiment("custom_exp", ["env_gnn", "merger_gnn"])

# Access specific fold predictions
fold_0_env = tracker.load_predictions("env_gnn", fold=0)
fold_0_merger = tracker.load_predictions("merger_gnn", fold=0)
```

## Scientific Interpretation

### Key Questions Addressed
1. **Assembly vs Environment**: Which provides better galaxy property predictions?
2. **Complementarity**: Do merger trees add information beyond environment?
3. **Physical Insights**: What features drive prediction errors?

### Expected Results Format
- **Performance metrics**: MSE, MAE, R², Pearson correlation by model and fold
- **Prediction files**: Include subhalo IDs, features, targets, and metadata
- **Residual analysis**: Environment model residuals for merger tree training
- **Feature correlations**: Which halo properties correlate with prediction errors

### Multi-Output Analysis
Results saved with separate columns:
- `pred_{model}_Mstar`, `pred_{model}_Mgas`: Model predictions
- `target_Mstar`, `target_Mgas`: True values
- Enables separate analysis of stellar vs gas mass predictions

## Environment Setup

**Recommended Environment**: Use the `pyg` conda environment for full functionality:
```bash
conda activate pyg
```

**PyTorch/CUDA Issues**: If you encounter CUDA library errors (`ncclCommRegister`), the experiment tracker will still work for analysis tasks (it only needs PyTorch for model saving during training).

**Dependencies**: The experiment tracker now gracefully handles missing PyTorch for analysis-only workflows.

## Configuration

All hyperparameters centralized in `src/config.py`:
- Model architectures (hidden dimensions, layers, etc.)
- Training parameters (learning rates, epochs, batch sizes)
- Data filtering (minimum masses, central vs satellite selection)
- Cross-validation setup (K=5 folds by default)

## Common Debugging

### Data Issues
- Check `TNG_BASE_PATH` and `TNG_EXTENDED_TREE_PATH` environment variables
- Ensure `illustris_data/` directory exists with TNG100-1 data
- Run `prepare_all_data()` manually if needed

### Training Issues
- Check GPU availability with `get_device()` from `training_utils`
- Monitor training logs in `results/experiment_{name}/logs/`
- Adjust batch sizes in `src/config.py` if memory issues

### Results Issues
- Use `--evaluate-only` flag to recompute metrics without retraining
- Check `metadata.json` for experiment configuration
- Verify consistent splits in `data_splits.pkl`

## Git Workflow

**Important**: When creating commits, NEVER include "Claude Code" in commit messages. Write descriptive, technical commit messages that focus on the actual changes made.

Good commit message examples:
- "Fix multi-output prediction merging in experiment tracker"
- "Add conditional PyTorch imports for analysis-only workflows"
- "Update documentation with environment setup instructions"

## Recent Updates (from git history)
- Fixed NaN handling in merger tree sizes (bde34a6)
- Added multi-output prediction capability (2abf0f0, 371c2a8)
- Implemented comprehensive experiment tracking (6e448a1)
- Fixed PyTorch Geometric compatibility (b1aedcd)
- Consolidated training utilities (1edefb8)