import h5py
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add

boxsize = (75 / 0.6774)     # box size in comoving Mpc/h
h = 0.6774                  # reduced Hubble constant
snapshot = 99

def make_cosmic_graph(subhalos: pd.DataFrame, D_link: int, periodic: bool=True) -> Data:
    """Produces an environmental (3d) graph from a subhalo catalog with some fixed linking length
    """
    df = subhalos.copy()

    subhalo_id = torch.tensor(df.index.values, dtype=torch.long)

    df.reset_index(drop=True)

    # DMO only properties
    x = torch.tensor(df[['subhalo_loghalomass_DMO', 'subhalo_logvmax_DMO']].values, dtype=torch.float)

    # hydro properties
    x_hydro = torch.tensor(df[["subhalo_loghalomass", 'subhalo_logvmax']].values, dtype=torch.float)

    # hydro total stellar mass
    y = torch.tensor(df[['subhalo_logstellarmass', 'subhalo_loggasmass', 'subhalo_logsfr', 'subhalo_gasmetallicity', 'subhalo_starmetallicity']].values, dtype=torch.float)

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

    # optimized way to compute overdensity
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    overdensity = torch.log10(
        scatter_add(
            10**(x.clone().detach()[source_nodes, 0]), 
            index=torch.tensor(target_nodes), 
            dim=0, 
            dim_size=len(x)
        )
    )

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=y,
        pos=pos,
        vel=vel,
        is_central=is_central,
        x_hydro=x_hydro,
        pos_hydro=pos_hydro,
        vel_hydro=vel_hydro,
        halfmassradius=halfmassradius,
        subhalo_id=subhalo_id,
        overdensity=overdensity,
    )

    return data

def make_merger_tree_graphs(trees: pd.DataFrame, subhalos: pd.DataFrame) -> list[Data]:
    """Use merger trees and z=0 subhalo catalog to create graphs of
    merger trees with final subhalo stellar mass as estimate
    """

    graphs = []

    subhalos.set_index(["subhalo_id_DMO"], inplace=True)
    all_subhalo_id_DMO = set(subhalos.index.values) 

    N_not_in_subhalo_cat = 0
    N_is_satellite = 0
    N_bad_stellar_mass = 0

    for root_descendent_id, tree in trees.groupby("root_descendent_id"):

        root_subhalo_id = tree.set_index("subhalo_tree_id").loc[root_descendent_id, "subhalo_id_in_this_snapshot"].astype(int)

        if root_subhalo_id not in all_subhalo_id_DMO:
            N_not_in_subhalo_cat += 1
            continue
        root_subhalo = subhalos.loc[root_subhalo_id]

        if not root_subhalo.is_central:
            N_is_satellite += 1
            continue

        if not np.isfinite(root_subhalo.subhalo_logstellarmass):
            N_bad_stellar_mass += 1
            continue

        x = torch.tensor(tree[["subhalo_loghalomass_DMO", "snapshot"]].values, dtype=torch.float)
        
        edges = [(s, d) for s, d in zip(tree.subhalo_tree_id.values, tree.descendent_id.values) if d != -1]
        
        # edges represented as subhalo_tree_ids (not needed)
        # edge_ids = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # edges represented as incrementing indices (reset for each tree)
        id2idx = {id: idx for id, idx in zip(tree.subhalo_tree_id.values, np.arange(len(tree)))}
        edge_index = torch.tensor(([[id2idx[e1], id2idx[e2]] for (e1, e2) in edges]), dtype=torch.long).t().contiguous()
        
        final_logstellarmass = torch.tensor([root_subhalo["subhalo_logstellarmass"]], dtype=torch.float)

        # TODO - remove snapshot number as node feature, input differences as edge features
        graph = Data(x=x, edge_index=edge_index, y=final_logstellarmass)
        graphs.append(graph)

    print(
        N_not_in_subhalo_cat,
        N_is_satellite,
        N_bad_stellar_mass
    )
    return graphs