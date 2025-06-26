# Assembly versus Environment

This project investigates the relative importance of halo assembly history versus local environment in determining galaxy properties. Using cosmological simulations from IllustrisTNG, we employ Graph Neural Networks (GNNs) to model the complex relationships between galaxy formation processes and compare the predictive power of merger history versus environmental factors.

## Motivation

Understanding the connection between dark matter subhalos and galaxies is a fundamental question in astrophysics. Galaxy evolution is driven by two important factors:

- **Assembly History**: The hierarchical build-up of dark matter halos through mergers and accretion over cosmic time
- **Environment**: The local density and tidal effects from neighboring structures

This project uses machine learning methods to quantitatively compare these "representations" by training models that predict galaxy properties (stellar mass, gas mass) from either merger trees or environmental graphs constructed from IllustrisTNG simulations. 

## Getting Started

### Data Requirements

This project requires IllustrisTNG simulation data ([see IllustrisTNG website](https://www.tng-project.org/data/)):

- **TNG data path**: `illustris_data/TNG100-1/`
- **Required data**: TNG100-1 hydrodynamic and dark matter-only simulations (snapshot 99, z=0)
- **Merger trees**: SubLink merger tree catalogs from TNG100-1-Dark


### Running the Analysis

The complete analysis workflow is demonstrated through Jupyter notebooks in the `notebook/` directory:

#### **01 Load and visualize data.ipynb**
- Loads and processes TNG subhalo catalogs and merger trees
- Creates environmental graphs (cosmic web with 3 Mpc linking length)  
- Generates merger tree graphs for assembly history analysis
- Produces data visualizations and saves intermediate datasets
- **Outputs**: `results/subhalos.parquet`, `results/cosmic_graphs_3Mpc.pkl`, `results/merger_trees.pkl`

#### **02 Train MLP.ipynb**
- Trains baseline Multi-Layer Perceptron models using basic subhalo properties
- **Input features**: subhalo mass, Vmax, central/satellite flag
- **Targets**: log stellar mass, log gas mass
- Uses 3-fold spatial cross-validation to avoid overfitting
- **Outputs**: `results/predictions/mlp_fold_{k}.parquet`, training logs, model weights

#### **03 Train Env GNN.ipynb**
- Trains Graph Neural Networks on environmental graphs
- Uses `EdgeInteractionGNN` architecture with explicit edge features
- Models local density and tidal interactions between neighboring subhalos
- **Outputs**: `results/predictions/env_gnn_fold_{k}.parquet`, training logs, model weights

#### **04 Train Merger Tree GNNs.ipynb**
- Trains GNNs on merger tree graphs to capture assembly history
- Uses `MultiSAGENet` architecture optimized for hierarchical tree structures
- Models the full merger and accretion history of each halo
- **Outputs**: `results/predictions/tree_gnn_fold_{k}.parquet`, training logs, model weights

#### **05 Comparing across models.ipynb**
- Aggregates results from all model types and cross-validation folds
- Computes comparative metrics (R², RMSE, MAE) for model performance
- Generates publication-quality comparison plots and training curves
- **Outputs**: `results/figures/results_pred-vs-true.pdf`, `results/figures/results_validation-rmse-curves.pdf`

### File Organization

The project generates several intermediate files and results:

```
results/
├── subhalos.parquet           # Processed subhalo catalog
├── cosmic_graphs_3Mpc.pkl     # Environmental graph dataset  
├── merger_trees.pkl           # Merger tree graph dataset
├── predictions/               # Model predictions for each fold
│   ├── mlp_fold_{0,1,2}.parquet
│   ├── env_gnn_fold_{0,1,2}.parquet
│   └── tree_gnn_fold_{0,1,2}.parquet
├── logs/                      # Training logs
│   ├── mlp_fold_{0,1,2}.txt
│   ├── env_gnn_fold_{0,1,2}.txt
│   └── tree_gnn_fold_{0,1,2}.txt
├── models/                    # Saved model weights
└── figures/                   # Generated plots and figures
```

## Data Details

**Source**: IllustrisTNG simulation suite, snapshot 99 (z=0)

**Environmental GNN**:
- TNG100-1 hydrodynamic simulation + TNG100-1-Dark (dark matter only)
- Subhalo selection: $M_{\rm halo} \geq 10^{10} M_{\odot}$, `SubhaloFlag == 0`
- Environmental graphs: 3 Mpc linking length for neighbor identification
- ~133k subhalos with complete environmental information

**Merger Tree GNN**:
- TNG100-1-Dark merger trees (SubLink catalogs)
- Merger tree reconstruction from z=0 back to formation
- ~123k subhalos with complete merger tree information
- Tree sizes range from single nodes to $>10^6$ progenitors

**Targets**: Log stellar mass and log gas mass for central galaxies with $M_{\bigstar} > 10^{8.5} M_{\odot}$

## Software Environment

### Dependencies

Required Python packages:
- **Core**: `numpy`, `pandas`, `scipy`, `matplotlib`
- **Astronomy**: `astropy`, `h5py`, `illustris_python`
- **Machine Learning**: `torch`, `torch_geometric`
- **Visualization**: `networkx`

Install with:
```bash
pip install numpy pandas scipy matplotlib h5py astropy networkx torch torch_geometric illustris_python
```

### Code Structure

The `src/` directory contains modular Python scripts:

- **`data.py`**: IllustrisTNG data loading, processing, and filtering
- **`loader.py`**: Graph dataset construction for PyTorch Geometric
- **`model.py`**: GNN model architectures (`EdgeInteractionGNN`, `MultiSAGENet`, `SAGEGraphConvNet`)  
- **`visualize.py`**: Merger tree visualization and plotting functions

## Model Architectures

**EdgeInteractionGNN**: Incorporates edge features in message passing for environmental modeling, following [Wu & Jespersen (2023)](https://arxiv.org/abs/2306.12327) and [Wu et al. (2024)](https://arxiv.org/abs/2402.07995)

**MultiSAGENet**: Multi-layer GraphSAGE architecture for hierarchical merger trees, based on [Jespersen et al. (2022)](https://arxiv.org/abs/2210.13473)

**Baseline MLP**: Standard feedforward network using only basic subhalo properties for comparison
