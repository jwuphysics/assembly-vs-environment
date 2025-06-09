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
Train models to predict the residuals from base model predictions:
```bash
python run_experiments.py --experiment "comparison" --models env_gnn
python run_experiments.py --experiment "comparison" --residual --base-model env_gnn
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

**Assembly History:** To reconstruct halo assembly histories, we use:
- IllustrisTNG TNG100-1-Dark merger trees (specifically `SubLink` catalogs).
We select halos with $M_{\rm halo} \geq 10^{10} M_{\odot}$ at z=0 to trace their merger histories.

## Code Structure

The main code for this project is in the `src/` directory. This directory contains Python scripts for data processing, model definition, and visualization.

Key Python files in `src/` include:
*   `data.py`: Handles loading, processing, and initial filtering of subhalo catalogs and merger trees from the IllustrisTNG simulations.
*   `loader.py`: Constructs graph-structured datasets (environmental graphs and merger tree graphs) compatible with PyTorch Geometric, using the processed data from `data.py`.
*   `model.py`: Defines the Graph Neural Network (GNN) architectures used in the project, such as `EdgeInteractionGNN`, `SAGEGraphConvNet`, and `MultiSAGENet`.
*   `visualize.py`: Provides functions for creating visualizations, primarily for plotting merger trees.

The project relies on several key Python libraries and frameworks, including:
*   `astropy`
*   `h5py`
*   `illustris_python`
*   `matplotlib`
*   `networkx`
*   `numpy`
*   `pandas`
*   `scipy`
*   `torch`
*   `torch_geometric`

## Models

This project uses Graph Neural Networks (GNNs) to analyze the graph-structured data generated by `loader.py`. These models are defined in `model.py` and are designed to learn from the complex relationships within environmental graphs and merger trees.

The specific GNN architectures include:
*   `EdgeInteractionGNN`: A GNN that explicitly incorporates edge features in its message passing steps, allowing for detailed modeling of interactions between connected nodes (subhalos or progenitors). This approach is particularly suited for capturing environmental influences, building on work such as [Wu & Jespersen (2023)](https://arxiv.org/abs/2306.12327) and [Wu et al. (2024)](https://arxiv.org/abs/2402.07995).
*   `SAGEGraphConvNet`: A GNN based on the GraphSAGE architecture, which learns by sampling and aggregating features from a node's local neighborhood.
*   `MultiSAGENet`: A multi-layer version of the SAGEGraphConvNet, potentially allowing for learning more complex representations, especially for deep merger trees (e.g., as explored in [Jespersen et al. (2022)](https://arxiv.org/abs/2210.13473)).

These models are applied to the environmental and merger tree graphs to predict galaxy properties and investigate the influence of assembly history and environment.

## Getting Started

This section provides guidance on setting up the project and running the code.

### Prerequisites/Dependencies

The project requires Python and a number of standard scientific computing libraries. Key dependencies include:
*   `numpy`
*   `pandas`
*   `scipy`
*   `matplotlib`
*   `h5py`
*   `astropy`
*   `networkx`
*   `torch`
*   `torch_geometric`

These can generally be installed using `pip` or `conda`. For example:
```bash
pip install numpy pandas scipy matplotlib h5py astropy networkx torch torch_geometric
```
Ensure you have a compatible version of PyTorch installed for your system, as PyTorch Geometric relies on it.

### Data Setup

An important step is to obtain the IllustrisTNG simulation data. This project expects data to be in a specific directory structure relative to the project root.
*   **Expected base path:** `illustris_data/TNG100-1/`
*   This path should contain the output files from the TNG100-1 simulation, including subhalo catalogs and merger trees (e.g., `SubLink` data).
*   For details on accessing and downloading IllustrisTNG data, please refer to the [IllustrisTNG data access website](https://www.tng-project.org/data/).

The scripts in `src/`, particularly `src/data.py`, are configured to load data from this path.

### Running the Code

Scripts in the `src/` directory are for different stages of the data processing and analysis pipeline:
*   `src/data.py`: This script is responsible for the initial loading of raw IllustrisTNG data (subhalo catalogs, merger trees), performing necessary processing, and applying initial filters.
*   `src/loader.py`: Takes the processed data from `data.py` and constructs graph-structured datasets. These datasets (environmental graphs and merger tree graphs) are formatted for use with PyTorch Geometric.
*   `src/model.py`: Contains the definitions of the GNN architectures (`EdgeInteractionGNN`, `SAGEGraphConvNet`, `MultiSAGENet`). These models are imported and used by other scripts for training and evaluation tasks.
*   `src/visualize.py`: Provides functions for creating visualizations, such as plotting merger trees, which can be useful for inspecting the data and model outputs.

There isn't a single main execution script. Users would typically import functions from or run these scripts sequentially, depending on their specific analysis goals:
1.  Process raw simulation data using functionalities from `src/data.py`.
2.  Generate graph datasets using `src/loader.py` with the processed data.
3.  Utilize the models from `src/model.py` for training, inference, or further analysis on the graph data.
4.  Employ `src/visualize.py` for inspecting data or results.
Refer to the individual scripts for more specific usage details.

## Visualization

The `src/visualize.py` script provides functions for creating visualizations related to the project's data, particularly focusing on merger trees.

### Merger Tree Plots

A key visualization capability is the plotting of dark matter halo merger trees. The primary function for this is `plot_merger_tree` within `src/visualize.py`.

These plots illustrate the hierarchical build-up of a selected dark matter halo over cosmic time. Each node in the tree represents a progenitor halo, and edges connect progenitors to their descendants. Nodes are typically colored or sized based on halo properties like mass or stellar mass, visually representing how halos assemble.

These visualizations are useful for:
*   Understanding the assembly history component of the research.
*   Debugging data processing steps related to merger trees.
*   Qualitatively inspecting the features that the GNN models might learn.

### Usage

To generate these plots, users can import the `plot_merger_tree` function from `src/visualize.py`. This function typically requires:
*   A processed merger tree, often as a pandas DataFrame (e.g., obtained from `data.py`).
*   The ID of the main halo whose tree is to be plotted.
*   Optionally, parameters to control node coloring, sizing, and layout.

The `plot_merger_tree` function includes a `save_figname` parameter, allowing plots to be saved to disk (e.g., as PDF or PNG files) for inclusion in presentations or publications.
