from astropy.io import fits
import glob
import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle
import random
import scipy
import tqdm



ROOT = Path(__file__).parent.parent.resolve()
tng_base_path = f"{ROOT}/illustris_data/TNG100-1"

seed = 255
rng = np.random.RandomState(seed)
random.seed(seed)


cuts = {
    "minimum_log_halo_mass": 10,
}

boxsize = (75 / 0.6774)     # box size in comoving Mpc/h
h = 0.6774                  # reduced Hubble constant
snapshot = 99


def load_subhalos(hydro: bool=True, cuts: dict=cuts, snapshot: int=snapshot) -> pd.DataFrame:
    """Load in TNG FoF_subfind subhalo catalog with relevant columns
    """
    
    use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_loghalomass', 'subhalo_logvmax'] 
    y_cols = ['subhalo_logstellarmass', 'subhalo_loggasmass', 'subhalo_logsfr', 'subhalo_gasmetallicity', 'subhalo_starmetallicity']

 
    base_path = (tng_base_path.replace("-1", "-1-Dark") if not hydro else tng_base_path) + "/output"

    subhalo_fields = [
        "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
        "SubhaloVel", "SubhaloVmax", "SubhaloGrNr", "SubhaloMassInMaxRadType",
        "SubhaloSpin", "SubhaloVelDisp", "SubhaloVmaxRad"
    ]

    if hydro:
        subhalo_fields += ["SubhaloFlag", "SubhaloSFR", "SubhaloGasMetallicity", "SubhaloStarMetallicity"]

    subhalos = il.groupcat.loadSubhalos(base_path, snapshot, fields=subhalo_fields) 

    pos = subhalos["SubhaloPos"][:,:3]

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3)
    subhalo_gasmass = subhalos["SubhaloMassType"][:,0]
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_halomass = subhalos["SubhaloMassType"][:,1]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4] 
    subhalo_vel = subhalos["SubhaloVel"][:]
    subhalo_vmax = subhalos["SubhaloVmax"][:]
    subhalo_flag = subhalos["SubhaloFlag"][:] if hydro else np.ones_like(subhalo_halomass) # note dummy values of 1 if DMO
    halo_id = subhalos["SubhaloGrNr"][:].astype(int)

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:].astype(int)
    group_pos = halos["GroupPos"][:] / (h*1e3)
    group_vel = halos["GroupVel"][:]

    subhalo_R50 = subhalos["SubhaloHalfmassRadType"][:,1]
    subhalo_MRmax = subhalos["SubhaloMassInMaxRadType"][:,1]
    subhalo_spin = (subhalos["SubhaloSpin"][:,0]**2 + subhalos["SubhaloSpin"][:,1]**2 + subhalos["SubhaloSpin"][:,2]**2)**0.5
    subhalo_Vdisp = subhalos["SubhaloVelDisp"][:]
    subhalo_VmaxRad = subhalos["SubhaloVmaxRad"][:]

    halos = pd.DataFrame(
        np.column_stack((
            np.arange(len(halo_mass)), group_pos, group_vel, halo_mass, halo_primarysubhalo
        )),
        columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mass', 'halo_primarysubhalo']
    )
    halos['halo_id'] = halos['halo_id'].astype(int)
    halos.set_index("halo_id", inplace=True)
    
    # get subhalos/galaxies      
    df = pd.DataFrame(
        np.column_stack([
            halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, 
            subhalo_n_stellar_particles, subhalo_gasmass, subhalo_stellarmass, subhalo_halomass, 
            subhalo_stellarhalfmassradius, subhalo_vmax, subhalo_R50, subhalo_MRmax, subhalo_spin, 
            subhalo_Vdisp, subhalo_VmaxRad,
        ]), 
        columns=[
            'halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 
            'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_gasmass', 'subhalo_stellarmass', 
            'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax', 'subhalo_R50', 'subhalo_MRmax', 
            'subhalo_spin', 'subhalo_Vdisp', 'subhalo_VmaxRad',
        ],
    )
    df["is_central"] = (halos.loc[df.halo_id]["halo_primarysubhalo"].values == df["subhalo_id"].values)
    
    # DMO-hydro matching
    if hydro:
        dmo_hydro_match = h5py.File(f"{tng_base_path}/postprocessing/subhalo_matching_to_dark.hdf5")
        df["subhalo_l_halo_tree"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_LHaloTree"]
        df["subhalo_sublink"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_SubLink"]      

        df["subhalo_sfr"] = subhalos["SubhaloSFR"][:]
        df["subhalo_gasmetallicity"] = subhalos["SubhaloGasMetallicity"][:] / 0.0127    # in solar metallicity units
        df["subhalo_starmetallicity"] = subhalos["SubhaloStarMetallicity"][:] / 0.0127  # in solar metallicity units


    # only drop in hydro
    if hydro:
        df = df[df["subhalo_flag"] != 0].copy()
    df['halo_id'] = df['halo_id'].astype(int)
    df['subhalo_id'] = df['subhalo_id'].astype(int)

    df["subhalo_logstellarmass"] = np.log10(df["subhalo_stellarmass"] / h)+10
    df["subhalo_loghalomass"] = np.log10(df["subhalo_halomass"] / h)+10
    df["subhalo_logvmax"] = np.log10(df["subhalo_vmax"])
    df["subhalo_logstellarhalfmassradius"] = np.log10(df["subhalo_stellarhalfmassradius"])
    df["subhalo_loggasmass"] = np.log10(df["subhalo_gasmass"])

    df["subhalo_logR50"] = np.log10(1+df["subhalo_R50"])
    df["subhalo_logMRmax"] = np.log10(df["subhalo_MRmax"] / h) + 10
    df["subhalo_logVdisp"] = np.log10(1+df["subhalo_Vdisp"])
    df["subhalo_logVmaxRad"] = np.log10(1+df["subhalo_VmaxRad"])
    df["subhalo_logspin"] = np.log10(1+df["subhalo_spin"])

    
    if hydro:
        df["subhalo_logsfr"] = np.log10(df["subhalo_sfr"])
        df["subhalo_gasmetallicity"] = np.log10(df["subhalo_gasmetallicity"]) + 12
        df["subhalo_starmetallicity"] = np.log10(df["subhalo_starmetallicity"]) + 12

    # drop non-log columns
    df.drop(
        [
            'subhalo_gasmass', 'subhalo_stellarmass', 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax',
            'subhalo_R50', 'subhalo_MRmax', 'subhalo_Vdisp', 'subhalo_VmaxRad', 'subhalo_spin' 
        ], 
        axis=1, 
        inplace=True
    )

    if hydro:
        # DM cuts -- do not put outside `if` statement, otherwise indices will be missing during the `prepare_subhalos()` call
        df.drop("subhalo_flag", axis=1, inplace=True)
        df = df[df["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()
        # stellar mass and particle cuts
        # df = df[df["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
        # df = df[df["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()
        
        # more columns to drop
        df.drop(
            ['subhalo_sfr'], 
            axis=1, 
            inplace=True
        )

    return df

def prepare_subhalos(cuts: dict=cuts) -> pd.DataFrame:
    """Helper function to load and join subhalos from dark matter only (DMO)
    and hydrodynamic simulations.

    Note that we force LHaloTree and sublink DMO linking methods to match.
    """
    
    subhalos = load_subhalos(hydro=True, cuts=cuts)
    subhalos_dmo = load_subhalos(hydro=False, cuts=cuts)

    valid_idxs_l_halo_tree = subhalos_dmo.index.isin(subhalos.subhalo_l_halo_tree)
    subhalos_linked = pd.concat(
        [
            (
                subhalos_dmo
                .loc[subhalos.subhalo_l_halo_tree[subhalos.subhalo_l_halo_tree != -1]]
                .reset_index(drop=True)
                .rename({c: c+"_DMO" for c in subhalos_dmo.columns}, axis=1)
            ),
            subhalos[subhalos.subhalo_l_halo_tree != -1].reset_index(drop=True),
        ], 
        axis=1,
    )

    # force l_halo_tree and sublink to match
    subhalos_linked = subhalos_linked[subhalos_linked.subhalo_l_halo_tree == subhalos_linked.subhalo_sublink].copy()
    
    # reiterate halo mass cuts just in case DMO masses are different...
    subhalos_linked = subhalos_linked[subhalos_linked.subhalo_loghalomass_DMO > cuts["minimum_log_halo_mass"]].copy()
    subhalos_linked.dropna(inplace=True, axis=0)

    return subhalos_linked

def load_trees(cuts=cuts) -> pd.DataFrame:
    """Load in all merger tree catalogs from dark matter only simulation, subject to optional mass cuts.
    """

    keys = [
        "SubhaloID", "SnapNum", "SubfindID", "DescendantID", "RootDescendantID", "SubhaloMass", "SubhaloVmax", 
        "GroupFirstSub", "SubhaloVelDisp", "SubhaloVmaxRad"
    ]

    list_of_trees = []

    for i in range(0, 9):
        filename = f"/home/john/research/illustris/TNG100-1/files/sublink/tree_extended.{i}.hdf5"
    
        with h5py.File(filename, 'r') as f:
            df = pd.DataFrame(
                {key: f[key][:] for key in keys}
            )
            df["subhalo_R50"] = f["SubhaloHalfmassRadType"][:,1]
            df["subhalo_MRmax"] = f["SubhaloMassInMaxRadType"][:,1]
            df["subhalo_spin"] = (f["SubhaloSpin"][:,0]**2 + f["SubhaloSpin"][:,1]**2 + f["SubhaloSpin"][:,2]**2)**0.5
    
        # rename and process columns -- note e.g. that what we call "subhalo_id_in_current_snapshot" is actually 
        # "SubfindID" and indexes into the subhalo group catalog at a given snapshot. For snapshot 99 in the 
        # environmental graph, we call this "subhalo_id"
    
        df.rename(
            {
                "SubfindID": "subhalo_id_in_this_snapshot",
                "DescendantID": "descendent_id",
                "RootDescendantID": "root_descendent_id",
                "SnapNum": "snapshot",
                "SubhaloID": "subhalo_tree_id",
                "GroupFirstSub": "is_central",
                "SubhaloVelDisp": "subhalo_Vdisp",
                "SubhaloVmaxRad": "subhalo_VmaxRad",
            },
            axis=1,
            inplace=True
        )
    
        df["subhalo_loghalomass_DMO"] = np.log10(df["SubhaloMass"]) + 10
        df["subhalo_logvmax_DMO"] = np.log10(1+df["SubhaloVmax"])
        
        df["subhalo_logR50"] = np.log10(1+df["subhalo_R50"])
        df["subhalo_logMRmax"] = np.log10(df["subhalo_MRmax"] / h) + 10
        df["subhalo_logVdisp"] = np.log10(1+df["subhalo_Vdisp"])
        df["subhalo_logVmaxRad"] = np.log10(1+df["subhalo_VmaxRad"])
        df["subhalo_logspin"] = np.log10(1+df["subhalo_spin"])
        
        df.drop(
            ["SubhaloMass", "SubhaloVmax", 'subhalo_R50', 'subhalo_MRmax', 'subhalo_Vdisp', 'subhalo_VmaxRad', 'subhalo_spin'], 
            axis=1, 
            inplace=True
        )

        # apply mass cuts *only on root subhalo*
        if cuts is not None:
            is_massive_root = (
                (df["descendent_id"] == -1)
                & (df["subhalo_loghalomass_DMO"] >= cuts["minimum_log_halo_mass"])
            )
    
            will_become_massive = df["root_descendent_id"].isin(df["subhalo_tree_id"][is_massive_root])
            list_of_trees.append(df[will_become_massive])
        else:
            list_of_trees.append(df)
        
    trees = pd.concat(list_of_trees, axis=0)
    trees.dropna(inplace=True, axis=0)
    return trees