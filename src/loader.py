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

def prepare_all_data():
    """Prepare all data sets: merger trees, crossmatched subhalo catalog, and env graph"""
    
    # create merger tree data set for crossmatching consistently...
    trees_fname = f"{results_dir}/merger_trees.pkl"
    if not os.path.exists(trees_fname):
        print("loading subhalos...")
        subhalos = prepare_subhalos()

        print("loading trees...")
        trees = load_trees()

        # some broken values in here...
        subhalos = subhalos[~np.isinf(subhalos.subhalo_logMRmax.values)].copy()

        print("making merger tree graphs...")
        merger_trees = make_merger_tree_graphs(trees, subhalos)

        with open(trees_fname, 'wb') as f:
            pickle.dump(merger_trees, f)
        print(f"saved ---> {trees_fname}!")
    else:
        with open(trees_fname, "rb") as f:
            merger_trees = pickle.load(f)
    
    # create crossmatched dataset
    crossmatched_fname = f"{results_dir}/subhalos_in_trees.parquet"
    if not os.path.exists(crossmatched_fname):
        print("crossmatching and saving subhalos...")
        subhalos = prepare_subhalos()
        subhalos = subhalos[~np.isinf(subhalos.subhalo_logMRmax.values)].copy()

        # make sure subhalos.subhalo_id_DMO <--> tree.root_subhalo_id at z=0
        subhalo_ids = [t.root_subhalo_id for t in merger_trees]
        subhalos = subhalos.set_index("subhalo_id_DMO").loc[subhalo_ids].reset_index()
        
        subhalos.to_parquet(crossmatched_fname)

    # finally, create the environmental graphs (only those overlapping with merger trees)
    graphs_fname = f"{results_dir}/cosmic_graphs.pkl"
    if not os.path.exists(graphs_fname):
        subhalos = pd.read_parquet(crossmatched_fname)

        print("making cosmic graph...")
        cosmic_graph = make_cosmic_graph(subhalos, D_link=D_link, periodic=True)
        
        with open(graphs_fname, 'wb') as f:
            pickle.dump(cosmic_graph, f)
        print(f"saved ---> {graphs_fname}!")
