from collections import deque
import h5py
import networkx as nx
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_scatter import scatter_add
from typing import List

try:
    from .data import *
except ImportError:
    from data import *

np.seterr(divide='ignore')

ROOT = Path(__file__).parent.parent.resolve()
results_dir = ROOT / "results"

D_link = 3 # Mpc

boxsize = (75 / 0.6774)     # box size in comoving Mpc/h
h = 0.6774                  # reduced Hubble constant
snapshot = 99

def make_cosmic_graph(subhalos: pd.DataFrame, D_link: int, periodic: bool=True) -> Data:
    """Produces an environmental (3d) graph from a subhalo catalog with some fixed linking length
    """
    df = subhalos.copy()

    idx = torch.tensor(df.index.values, dtype=torch.long)
    subhalo_id = torch.tensor(df.subhalo_id_DMO.values, dtype=torch.long)

    # df.reset_index(drop=True)

    # DMO only properties
    x = torch.tensor(
        df[[
            'subhalo_loghalomass_DMO', 'subhalo_logvmax_DMO', 'subhalo_logR50', 'subhalo_logMRmax', 
            'subhalo_logVdisp', 'subhalo_logVmaxRad', 'subhalo_logspin',
        ]].values, 
        dtype=torch.float,
    )

    # hydro properties
    x_hydro = torch.tensor(
        df[['subhalo_loghalomass', 'subhalo_logvmax']].values, 
        dtype=torch.float,
    )

    # hydro total stellar mass and gas mass
    y = torch.tensor(df[['subhalo_logstellarmass', 'subhalo_loggasmass']].values, dtype=torch.float)

    # phase space coordinates
    pos = torch.tensor(df[['subhalo_x_DMO', 'subhalo_y_DMO', 'subhalo_z_DMO']].values, dtype=torch.float)
    vel = torch.tensor(df[['subhalo_vx_DMO', 'subhalo_vy_DMO', 'subhalo_vz_DMO']].values, dtype=torch.float)

    pos_hydro = torch.tensor(df[['subhalo_x', 'subhalo_y', 'subhalo_z']].values, dtype=torch.float)
    vel_hydro = torch.tensor(df[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].values, dtype=torch.float)

    is_central = torch.tensor(df[['is_central']].values, dtype=torch.int)
    halfmassradius = torch.tensor(df[['subhalo_logstellarhalfmassradius']].values, dtype=torch.float)

    # make links
    kd_tree = scipy.spatial.KDTree(pos, leafsize=25, boxsize=boxsize)
    edge_index = kd_tree.query_pairs(r=D_link, output_type="ndarray").astype(int)

    # normalize positions
    df[['subhalo_x', 'subhalo_y', 'subhalo_z']] = df[['subhalo_x', 'subhalo_y', 'subhalo_z']]/(boxsize/2)

    # add reverse pairs
    edge_index = to_undirected(torch.Tensor(edge_index).t().contiguous().type(torch.long)).numpy()

    # remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    # Write in pytorch-geometric format
    edge_index = edge_index.reshape((2,-1))
    num_pairs = edge_index.shape[1]

    row, col = edge_index

    diff = pos[row]-pos[col]
    dist = np.linalg.norm(diff, axis=1)

    use_gal = True

    if periodic:
        # Take into account periodic boundary conditions, correcting the distances
        for i, pos_i in enumerate(diff):
            for j, coord in enumerate(pos_i):
                if coord > D_link:
                    diff[i,j] -= boxsize  # Boxsize normalize to 1
                elif -coord > D_link:
                    diff[i,j] += boxsize  # Boxsize normalize to 1

    centroid = np.array(pos.mean(0)) # define arbitrary coordinate, invariant to translation/rotation shifts, but not stretches

    unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
    unitdiff = diff/dist.reshape(-1,1)
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:], unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:], unitdiff[i,:]) for i in range(num_pairs)])

    # same invariant edge features but for velocity
    velnorm = np.vstack(df[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].to_numpy())
    vel_diff = velnorm[row]-velnorm[col]
    vel_norm = np.linalg.norm(vel_diff, axis=1)
    vel_centroid = np.array(velnorm.mean(0))

    vel_unitrow = (velnorm[row]-vel_centroid)/np.linalg.norm(velnorm[row]-vel_centroid, axis=1).reshape(-1, 1)
    vel_unitcol = (velnorm[col]-vel_centroid)/np.linalg.norm(velnorm[col]-vel_centroid, axis=1).reshape(-1, 1)
    vel_unitdiff = vel_diff / vel_norm.reshape(-1,1)
    vel_cos1 = np.array([np.dot(vel_unitrow[i,:], vel_unitcol[i,:]) for i in range(num_pairs)])
    vel_cos2 = np.array([np.dot(vel_unitrow[i,:], vel_unitdiff[i,:]) for i in range(num_pairs)])

    # build edge features
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1), vel_norm.reshape(-1,1), vel_cos1.reshape(-1,1), vel_cos2.reshape(-1,1)], axis=1)

    # use loops here no matter what; can always remove later
    loops = np.zeros((2,pos.shape[0]),dtype=int)
    atrloops = np.zeros((pos.shape[0], edge_attr.shape[1]))
    for i, posit in enumerate(pos):
        loops[0,i], loops[1,i] = i, i
        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
    edge_index = np.append(edge_index, loops, 1)
    edge_attr = np.append(edge_attr, atrloops, 0)
    edge_index = edge_index.astype(int)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # optimized way to compute overdensity
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    overdensity = torch.log10(
        scatter_add(
            10**(x.clone().detach()[source_nodes, 0]), 
            index=target_nodes, 
            dim=0, 
            dim_size=len(x)
        )
    )

    # some distances and veldist are nan... remove them and remake graph
    
    masked = ~torch.isnan(edge_attr).any(1)
    
    edge_index = edge_index[:, masked]
    edge_attr = edge_attr[masked]
    
    new_nodes = torch.unique(edge_index)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_nodes.tolist())}
    
    edge_index_mapped = torch.tensor(
        [
            [
                node_mapping[idx.item()] for idx in edge
            ] 
            for edge in edge_index.t()
        ]
    ).t()

    # include is_central as node feature
    x = torch.cat([x, is_central.type(torch.float)], axis=1)[new_nodes]
    y = y[new_nodes]
    pos = pos[new_nodes]
    vel = vel[new_nodes]
    is_central = is_central[new_nodes]
    x_hydro = x_hydro[new_nodes]
    pos_hydro = pos_hydro[new_nodes]
    vel_hydro = vel_hydro[new_nodes]
    halfmassradius = halfmassradius[new_nodes]
    subhalo_id = subhalo_id[new_nodes]
    idx = idx[new_nodes]
    overdensity = overdensity[new_nodes]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
        vel=vel,
        is_central=is_central,
        x_hydro=x_hydro,
        pos_hydro=pos_hydro,
        vel_hydro=vel_hydro,
        halfmassradius=halfmassradius,
        subhalo_id=subhalo_id,
        idx=idx,
        overdensity=overdensity,
    )

    return data

def make_merger_tree_graphs(trees: pd.DataFrame, subhalos: pd.DataFrame) -> list[Data]:
    """Use merger trees and z=0 subhalo catalog to create graphs of
    merger trees with final subhalo stellar mass as estimate. 

    Note that subhalos catalog will impose additional cuts -- make sure 
    that these are consistent with the merger tree cuts!
    """

    graphs = []

    if subhalos.index.name not in ["subhalo_id_DMO", "subhalo_id"]:
        subhalos.set_index(["subhalo_id_DMO"], inplace=True)

    all_subhalo_id_DMO = set(subhalos.index.values) 

    N_not_in_subhalo_cat = 0
    N_bad_stellar_mass = 0

    for root_descendent_id, tree in trees.groupby("root_descendent_id"):

        root_subhalo_id = tree.set_index("subhalo_tree_id").loc[root_descendent_id, "subhalo_id_in_this_snapshot"].astype(int)

        if root_subhalo_id not in all_subhalo_id_DMO:
            N_not_in_subhalo_cat += 1
            continue
        root_subhalo = subhalos.loc[root_subhalo_id]

        if not np.isfinite(root_subhalo.subhalo_logstellarmass):
            N_bad_stellar_mass += 1
            continue

        x = torch.tensor(
            tree[[
                "subhalo_loghalomass_DMO", "subhalo_logvmax_DMO", 'subhalo_logR50', 'subhalo_logMRmax', 
                'subhalo_logVdisp', 'subhalo_logVmaxRad', 'subhalo_logspin', "snapshot"
            ]].values, 
            dtype=torch.float
        )

        is_central = root_subhalo.is_central.astype(int)
        
        edges = [(s, d) for s, d in zip(tree.subhalo_tree_id.values, tree.descendent_id.values) if d != -1]

        # edges represented as incrementing indices (reset for each tree)
        id2idx = {id: idx for id, idx in zip(tree.subhalo_tree_id.values, np.arange(len(tree)))}
        edge_index = torch.tensor(([[id2idx[e1], id2idx[e2]] for (e1, e2) in edges]), dtype=torch.long).t().contiguous()
        
        final_logstellarmass = torch.tensor([root_subhalo["subhalo_logstellarmass"]], dtype=torch.float)
        final_loggasmass = torch.tensor([root_subhalo["subhalo_loggasmass"]], dtype=torch.float)
        final_targets = torch.stack([final_logstellarmass[0], final_loggasmass[0]], dim=0).unsqueeze(0)

        graph = Data(
            x=x, 
            edge_index=edge_index, 
            y=final_targets, 
            is_central=is_central, 
            root_subhalo_id=root_subhalo_id
        )

        # get difference of snapshot numbers and add as edge feature
        graph.edge_attr = graph.x[graph.edge_index[0],-1] - graph.x[graph.edge_index[1],-1]
        graphs.append(graph)

    return graphs

def trim_merger_trees(graphs: List[Data]) -> List[Data]:
    """
    Remove all nodes in the merger tree graphs that only have one incoming and one outgoing edge,
    and update the edge_attr appropriately.
    """
    
    trimmed_graphs = []

    for graph in tqdm(graphs):

        # find nodes to be removed (have one incoming and one outgoing edge)
        edge_index = graph.edge_index
        row, col = edge_index
        
        deg_in = torch.zeros_like(graph.x[:, 0], dtype=torch.long)
        deg_in.scatter_add_(0, row, torch.ones_like(row))
        deg_out = torch.zeros_like(graph.x[:, 0], dtype=torch.long)
        deg_out.scatter_add_(0, col, torch.ones_like(col))
        nodes_to_remove = torch.logical_and(deg_in == 1, deg_out == 1)

        # skip if no nodes to remove
        if not nodes_to_remove.any():
            trimmed_graphs.append(graph)
            continue

        # redo indices in order of visited nodes
        sorted_nodes = _topological_sort(edge_index)
        visited_nodes = _walk_down_graph(
            graph.edge_index, 
            sorted_nodes
        )
        
        # incrementing indices of nodes to keep
        nodes_to_keep = torch.where(~nodes_to_remove)[0].flip(0)
        keep_mask = torch.zeros(edge_index.max() + 1, dtype=torch.bool)
        keep_mask[nodes_to_keep] = True
        
        trimmed_edge_index = _trim_graph(edge_index, keep_mask)

        # Re-map the nodes and edges
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep.tolist())}
        new_edge_index = torch.stack([
            torch.tensor([node_mapping[node.item()] for node in trimmed_edge_index[row]]) 
            for row in range(2)
        ])

        trimmed_graph = Data(
            x=graph.x[nodes_to_keep], 
            edge_index=new_edge_index, 
            root_subhalo_id=graph.root_subhalo_id, 
            y=graph.y,
        )

        # estimate snapshot differences again...
        trimmed_graph.edge_attr = (
            trimmed_graph.x[trimmed_graph.edge_index[1],-1] - 
            trimmed_graph.x[trimmed_graph.edge_index[0],-1]
        ).reshape(-1, 1)
        
        trimmed_graphs.append(trimmed_graph)
        

    return trimmed_graphs

def _topological_sort(edge_index):
    """Apply the networkx topological_sort for PyG Data object"""
    G = nx.DiGraph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    sorted_nodes = list(nx.topological_sort(G))
    return torch.tensor(sorted_nodes)
    
def _walk_down_graph(edge_index, sorted_nodes):
    """Sort edges in order of walking down the graph"""
    visited_nodes = []
    for node in sorted_nodes:
        # Find the edges where this node is a source
        outgoing_edge_indices = (edge_index[0] == node).nonzero(as_tuple=True)[0]
        for idx in outgoing_edge_indices:
            target_node = edge_index[1, idx]
            visited_nodes.append((node.item(), target_node.item()))
    
    return visited_nodes

def _trim_graph(edge_index, keep_mask):
    """For a directed acyclic graph specified by `edge_index` for which 
    one can "walk" down an edge to the next node, that is already in order,
    and also given a set of nodes that will be kept, `keep_mask`, then
    return a trimmed graph where the edges are recombined.
    """
    reduced_edges = []
    
    # represent edge_index as adjacency list
    adj_list = {i: [] for i in range(len(keep_mask))}
    for i, j in edge_index.t():
        adj_list[i.item()].append(j.item())

    # breadth-first search (BFS) using deque
    for start_node in torch.where(keep_mask)[0]:
        visited = torch.zeros_like(keep_mask)
        queue = deque([start_node.item()])
        
        # Track paths using a dictionary
        parent = {start_node.item(): None}
        
        while queue:
            current = queue.popleft()
            
            # Check if this node is a target node and stop if it's not the start node
            if keep_mask[current] and current != start_node.item():
                # Trace back the path to start_node and break
                path = []
                step = current
                while step is not None:
                    path.append(step)
                    step = parent[step]
                path.reverse()
                
                # Check if the path is valid and add the edge
                if len(path) > 1:
                    reduced_edges.append((path[0], path[-1]))
                break
            
            # Visit each neighbor
            for neighbor in adj_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
    return torch.tensor(reduced_edges).t() if reduced_edges else torch.empty((2, 0), dtype=torch.long)

def apply_consistent_cuts(subhalos: pd.DataFrame, 
                          minimum_stellar_mass: float = None,
                          only_centrals: bool = False) -> pd.DataFrame:
    """Apply consistent quality cuts to ensure reliable cross-model comparisons.
    
    Args:
        subhalos: Full crossmatched subhalo catalog
        minimum_stellar_mass: Minimum log stellar mass (default: no cut)
        only_centrals: If True, keep only central galaxies (default: False)
        
    Returns:
        Filtered subhalo catalog with consistent cuts applied
    """
    print(f"Starting with {len(subhalos)} crossmatched subhalos")
    
    # Remove subhalos with infinite/invalid values
    subhalos_clean = subhalos[~np.isinf(subhalos.subhalo_logMRmax.values)].copy()
    print(f"After removing inf logMRmax: {len(subhalos_clean)}")
    
    # Remove subhalos with invalid stellar mass
    subhalos_clean = subhalos_clean[np.isfinite(subhalos_clean.subhalo_logstellarmass)].copy()
    print(f"After removing invalid stellar mass: {len(subhalos_clean)}")
    
    # Apply stellar mass cut if specified
    if minimum_stellar_mass is not None:
        subhalos_clean = subhalos_clean[subhalos_clean.subhalo_logstellarmass > minimum_stellar_mass].copy()
        print(f"After stellar mass cut (>{minimum_stellar_mass}): {len(subhalos_clean)}")
    
    # Apply centrals-only filter if specified
    if only_centrals:
        subhalos_clean = subhalos_clean[subhalos_clean.is_central == 1].copy()
        print(f"After centrals-only cut: {len(subhalos_clean)}")
    
    return subhalos_clean

def make_merger_tree_graphs_consistent(trees: pd.DataFrame, subhalos_filtered: pd.DataFrame) -> list[Data]:
    """Create merger tree graphs using only the pre-filtered subhalo set.
    
    This ensures merger trees contain exactly the same subhalos as cosmic graphs.
    """
    # Set index for fast lookup
    if subhalos_filtered.index.name != "subhalo_id_DMO":
        subhalos_filtered = subhalos_filtered.set_index("subhalo_id_DMO")
    
    # Only create trees for subhalos in our filtered set
    valid_subhalo_ids = set(subhalos_filtered.index.values)
    
    graphs = []
    N_not_in_filtered_set = 0
    N_created = 0
    
    for root_descendent_id, tree in trees.groupby("root_descendent_id"):
        root_subhalo_id = tree.set_index("subhalo_tree_id").loc[root_descendent_id, "subhalo_id_in_this_snapshot"].astype(int)
        
        # Only include if this subhalo is in our filtered set
        if root_subhalo_id not in valid_subhalo_ids:
            N_not_in_filtered_set += 1
            continue
            
        root_subhalo = subhalos_filtered.loc[root_subhalo_id]
        
        # Handle case where root_subhalo might be a Series or DataFrame
        if hasattr(root_subhalo, 'iloc'):
            # If multiple rows returned, take the first one
            root_subhalo = root_subhalo.iloc[0] if len(root_subhalo.shape) > 1 else root_subhalo
        
        x = torch.tensor(
            tree[[
                "subhalo_loghalomass_DMO", "subhalo_logvmax_DMO", 'subhalo_logR50', 'subhalo_logMRmax', 
                'subhalo_logVdisp', 'subhalo_logVmaxRad', 'subhalo_logspin', "snapshot"
            ]].values, 
            dtype=torch.float
        )
        
        is_central = int(root_subhalo["is_central"])
        
        edges = [(s, d) for s, d in zip(tree.subhalo_tree_id.values, tree.descendent_id.values) if d != -1]
        
        # edges represented as incrementing indices (reset for each tree)
        id2idx = {id: idx for id, idx in zip(tree.subhalo_tree_id.values, np.arange(len(tree)))}
        edge_index = torch.tensor(([[id2idx[e1], id2idx[e2]] for (e1, e2) in edges]), dtype=torch.long).t().contiguous()
        
        final_logstellarmass = float(root_subhalo["subhalo_logstellarmass"])
        final_loggasmass = float(root_subhalo["subhalo_loggasmass"])
        final_targets = torch.tensor([[final_logstellarmass, final_loggasmass]], dtype=torch.float)
        
        graph = Data(
            x=x, 
            edge_index=edge_index, 
            y=final_targets, 
            is_central=is_central, 
            root_subhalo_id=root_subhalo_id
        )
        
        # get difference of snapshot numbers and add as edge feature
        graph.edge_attr = graph.x[graph.edge_index[0],-1] - graph.x[graph.edge_index[1],-1]
        graphs.append(graph)
        N_created += 1
    
    print(f"Created {N_created} merger trees from filtered subhalos")
    print(f"Skipped {N_not_in_filtered_set} trees not in filtered set")
    
    return graphs

def make_cosmic_graph_consistent(subhalos_filtered: pd.DataFrame) -> Data:
    """Create cosmic graph using the same filtered subhalo set as merger trees.
    
    This ensures consistent subhalo_id mappings across all models.
    """
    return make_cosmic_graph(subhalos_filtered, D_link=D_link, periodic=True)

def validate_dataset_consistency(cosmic_graph: Data, merger_trees: list[Data], subhalos_filtered: pd.DataFrame):
    """Validate that all datasets contain exactly the same set of subhalos.
    
    Raises AssertionError if inconsistencies are found.
    """
    # Get subhalo_id sets from each dataset
    cosmic_ids = set(cosmic_graph.subhalo_id.tolist())
    merger_ids = set([t.root_subhalo_id for t in merger_trees])
    filtered_ids = set(subhalos_filtered['subhalo_id_DMO'].values if 'subhalo_id_DMO' in subhalos_filtered.columns 
                      else subhalos_filtered.index.values)
    
    print(f"Cosmic graph subhalos: {len(cosmic_ids)}")
    print(f"Merger tree subhalos: {len(merger_ids)}")
    print(f"Filtered catalog subhalos: {len(filtered_ids)}")
    
    # Check for perfect consistency
    if cosmic_ids != merger_ids:
        only_cosmic = cosmic_ids - merger_ids
        only_merger = merger_ids - cosmic_ids
        raise AssertionError(
            f"Subhalo ID mismatch between cosmic graph and merger trees!\n"
            f"Only in cosmic: {len(only_cosmic)} subhalos\n"
            f"Only in merger: {len(only_merger)} subhalos\n"
            f"Sample cosmic-only IDs: {list(only_cosmic)[:10]}\n"
            f"Sample merger-only IDs: {list(only_merger)[:10]}"
        )
    
    # Validate that cosmic graph subhalos are subset of filtered catalog
    if not cosmic_ids.issubset(filtered_ids):
        missing = cosmic_ids - filtered_ids
        raise AssertionError(
            f"Cosmic graph contains {len(missing)} subhalos not in filtered catalog!\n"
            f"Sample missing IDs: {list(missing)[:10]}"
        )
    
    print("✓ All datasets contain consistent subhalo_id mappings")

def load_consistent_datasets():
    """Load all datasets with validation that they're consistent.
    
    Returns:
        Tuple of (cosmic_graph, merger_trees, subhalos_base)
        
    Note: 
        Returns the base consistent datasets without additional filtering.
        Apply model-specific filtering (stellar mass cuts, centrals-only, etc.) 
        at training/evaluation time, not at dataset loading time.
        
    Raises:
        AssertionError if datasets are inconsistent
        FileNotFoundError if datasets don't exist
    """
    import pickle
    
    # Check that all files exist
    cosmic_fname = f"{results_dir}/cosmic_graphs.pkl"
    trees_fname = f"{results_dir}/merger_trees.pkl"
    subhalos_fname = f"{results_dir}/subhalos_in_trees.parquet"
    
    if not all(os.path.exists(f) for f in [cosmic_fname, trees_fname, subhalos_fname]):
        raise FileNotFoundError(
            "Consistent datasets not found. Run prepare_all_data() first to generate them."
        )
    
    # Load all datasets
    with open(cosmic_fname, 'rb') as f:
        cosmic_graph = pickle.load(f)
    
    with open(trees_fname, 'rb') as f:
        merger_trees = pickle.load(f)
    
    subhalos_base = pd.read_parquet(subhalos_fname)
    subhalos_base = apply_consistent_cuts(subhalos_base)  # Only basic quality cuts
    
    # Validate consistency of base datasets
    validate_dataset_consistency(cosmic_graph, merger_trees, subhalos_base)
    
    return cosmic_graph, merger_trees, subhalos_base

def apply_model_specific_filtering(cosmic_graph, merger_trees, subhalos_base, 
                                  minimum_stellar_mass: float = None, only_centrals: bool = False):
    """Apply model-specific filtering to consistent datasets.
    
    Args:
        cosmic_graph: Cosmic graph data
        merger_trees: List of merger tree graphs
        subhalos_base: Base subhalo catalog
        minimum_stellar_mass: Minimum log stellar mass cut
        only_centrals: Keep only central galaxies
        
    Returns:
        Filtered versions of (cosmic_graph, merger_trees, subhalos_filtered) with consistent subhalo_ids
    """
    if minimum_stellar_mass is None and not only_centrals:
        # No additional filtering needed
        return cosmic_graph, merger_trees, subhalos_base
    
    print(f"Applying model-specific filtering: M*>{minimum_stellar_mass}, centrals_only={only_centrals}")
    
    # Apply filtering to subhalo catalog
    subhalos_filtered = apply_consistent_cuts(subhalos_base, minimum_stellar_mass, only_centrals)
    valid_subhalo_ids = set(subhalos_filtered['subhalo_id_DMO'].values)
    
    print(f"Filtering: {len(subhalos_base)} -> {len(subhalos_filtered)} subhalos")
    
    # Filter cosmic graph
    cosmic_mask = torch.tensor([sid.item() in valid_subhalo_ids for sid in cosmic_graph.subhalo_id])
    cosmic_filtered = Data(
        x=cosmic_graph.x[cosmic_mask],
        edge_index=None,  # Will need to rebuild edges for the filtered graph
        y=cosmic_graph.y[cosmic_mask],
        pos=cosmic_graph.pos[cosmic_mask],
        vel=cosmic_graph.vel[cosmic_mask],
        is_central=cosmic_graph.is_central[cosmic_mask],
        x_hydro=cosmic_graph.x_hydro[cosmic_mask],
        pos_hydro=cosmic_graph.pos_hydro[cosmic_mask],
        vel_hydro=cosmic_graph.vel_hydro[cosmic_mask],
        halfmassradius=cosmic_graph.halfmassradius[cosmic_mask],
        subhalo_id=cosmic_graph.subhalo_id[cosmic_mask],
        idx=cosmic_graph.idx[cosmic_mask],
        overdensity=cosmic_graph.overdensity[cosmic_mask],
    )
    
    # Rebuild edges for filtered cosmic graph
    cosmic_filtered = _rebuild_cosmic_graph_edges(cosmic_filtered, subhalos_filtered)
    
    # Filter merger trees
    trees_filtered = [tree for tree in merger_trees if tree.root_subhalo_id in valid_subhalo_ids]
    
    print(f"Filtered datasets: cosmic={len(cosmic_filtered.subhalo_id)}, trees={len(trees_filtered)}")
    
    return cosmic_filtered, trees_filtered, subhalos_filtered

def _rebuild_cosmic_graph_edges(cosmic_data, subhalos_filtered):
    """Rebuild edges and edge attributes for filtered cosmic graph using full graph construction."""
    # Create a temporary dataframe for full graph rebuilding
    # We need the full set of columns that make_cosmic_graph expects
    temp_df = pd.DataFrame({
        'subhalo_id_DMO': cosmic_data.subhalo_id.numpy(),
        'subhalo_x_DMO': cosmic_data.pos[:, 0].numpy(),
        'subhalo_y_DMO': cosmic_data.pos[:, 1].numpy(), 
        'subhalo_z_DMO': cosmic_data.pos[:, 2].numpy(),
        'subhalo_vx_DMO': cosmic_data.vel[:, 0].numpy(),
        'subhalo_vy_DMO': cosmic_data.vel[:, 1].numpy(),
        'subhalo_vz_DMO': cosmic_data.vel[:, 2].numpy(),
        'subhalo_loghalomass_DMO': cosmic_data.x[:, 0].numpy(),
        'subhalo_logvmax_DMO': cosmic_data.x[:, 1].numpy(),
        'subhalo_logR50': cosmic_data.x[:, 2].numpy(),
        'subhalo_logMRmax': cosmic_data.x[:, 3].numpy(),
        'subhalo_logVdisp': cosmic_data.x[:, 4].numpy(),
        'subhalo_logVmaxRad': cosmic_data.x[:, 5].numpy(),
        'subhalo_logspin': cosmic_data.x[:, 6].numpy(),
        'is_central': cosmic_data.is_central.numpy().flatten(),
        'subhalo_loghalomass': cosmic_data.x_hydro[:, 0].numpy(),
        'subhalo_logvmax': cosmic_data.x_hydro[:, 1].numpy(),
        'subhalo_x': cosmic_data.pos_hydro[:, 0].numpy(),
        'subhalo_y': cosmic_data.pos_hydro[:, 1].numpy(),
        'subhalo_z': cosmic_data.pos_hydro[:, 2].numpy(),
        'subhalo_vx': cosmic_data.vel_hydro[:, 0].numpy(),
        'subhalo_vy': cosmic_data.vel_hydro[:, 1].numpy(),
        'subhalo_vz': cosmic_data.vel_hydro[:, 2].numpy(),
        'subhalo_logstellarmass': cosmic_data.y[:, 0].numpy(),
        'subhalo_loggasmass': cosmic_data.y[:, 1].numpy(),
        'subhalo_logstellarhalfmassradius': cosmic_data.halfmassradius.numpy().flatten(),
    })
    
    # Rebuild the complete cosmic graph with edges and edge attributes
    rebuilt_graph = make_cosmic_graph(temp_df, D_link=D_link, periodic=True)
    
    # Update the original cosmic_data with properly built edges and edge_attr
    cosmic_data.edge_index = rebuilt_graph.edge_index
    cosmic_data.edge_attr = rebuilt_graph.edge_attr
    cosmic_data.overdensity = rebuilt_graph.overdensity  # Also update overdensity for filtered graph
    
    return cosmic_data

def prepare_all_data(minimum_stellar_mass: float = None, only_centrals: bool = False):
    """Prepare all data sets with consistent filtering for reliable cross-model comparisons.
    
    This function ensures that cosmic graphs and merger trees contain exactly the same
    set of subhalos by applying filtering in the correct order and validating consistency.
    
    Args:
        minimum_stellar_mass: Minimum log stellar mass filter (default: None for no cut)
        only_centrals: If True, keep only central galaxies (default: False for all galaxies)
        
    Usage:
        # For full comparison (all galaxies):
        prepare_all_data()
        
        # For merger-tree style analysis (centrals with M* > 8.5):
        prepare_all_data(minimum_stellar_mass=8.5, only_centrals=True)
    """
    
    # Step 1: Load and apply consistent filtering to crossmatched subhalos
    crossmatched_fname = f"{results_dir}/subhalos_in_trees.parquet"
    if not os.path.exists(crossmatched_fname):
        print("Creating crossmatched subhalo catalog...")
        subhalos = prepare_subhalos()
        subhalos.to_parquet(crossmatched_fname)
        print(f"Saved crossmatched catalog: {crossmatched_fname}")
    
    # Load the crossmatched catalog (single source of truth)
    print("Loading crossmatched subhalos...")
    subhalos = pd.read_parquet(crossmatched_fname)
    
    # Step 2: Apply consistent quality cuts to create master filtered set
    print("Applying consistent quality cuts...")
    subhalos_filtered = apply_consistent_cuts(subhalos, minimum_stellar_mass, only_centrals)
    print(f"Filtered subhalos: {len(subhalos)} -> {len(subhalos_filtered)}")
    
    # Step 3: Create merger tree dataset from filtered subhalos
    trees_fname = f"{results_dir}/merger_trees.pkl"
    if not os.path.exists(trees_fname):
        print("Loading raw merger trees...")
        trees = load_trees()
        
        print("Making merger tree graphs from filtered subhalos...")
        merger_trees = make_merger_tree_graphs_consistent(trees, subhalos_filtered)
        
        with open(trees_fname, 'wb') as f:
            pickle.dump(merger_trees, f)
        print(f"Saved merger trees: {trees_fname}")
    else:
        with open(trees_fname, "rb") as f:
            merger_trees = pickle.load(f)
    
    # Step 4: Create cosmic graph from same filtered subhalos
    graphs_fname = f"{results_dir}/cosmic_graphs.pkl"
    if not os.path.exists(graphs_fname):
        print("Making cosmic graph from filtered subhalos...")
        cosmic_graph = make_cosmic_graph_consistent(subhalos_filtered)
        
        with open(graphs_fname, 'wb') as f:
            pickle.dump(cosmic_graph, f)
        print(f"Saved cosmic graph: {graphs_fname}")
    else:
        with open(graphs_fname, "rb") as f:
            cosmic_graph = pickle.load(f)
    
    # Step 5: Validate consistency
    print("Validating data consistency...")
    validate_dataset_consistency(cosmic_graph, merger_trees, subhalos_filtered)
    
    print("✓ All datasets prepared with consistent subhalo_id mappings")
