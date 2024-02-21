

cuts = {
    "minimum_log_halo_mass": 10,
}

boxsize = (75 / 0.6774)     # box size in comoving Mpc/h
h = 0.6774                  # reduced Hubble constant
snapshot = 99


def load_subhalos(hydro=True, normalization_params=normalization_params, cuts=cuts, snapshot=snapshot):
    
    use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_loghalomass', 'subhalo_logvmax'] 
    y_cols = ['subhalo_logstellarmass']

 
    base_path = tng_base_path.replace("TNG300-1", "TNG300-1-Dark") if not hydro else tng_base_path

    subhalo_fields = [
        "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
        "SubhaloVel", "SubhaloVmax", "SubhaloGrNr", 
    ]

    if hydro:
        subhalo_fields += ["SubhaloFlag"]

    subhalos = il.groupcat.loadSubhalos(base_path, snapshot, fields=subhalo_fields) 

    pos = subhalos["SubhaloPos"][:,:3]

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3)
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_halomass = subhalos["SubhaloMassType"][:,1]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4]  / normalization_params["norm_half_mass_radius"]
    subhalo_vel = subhalos["SubhaloVel"][:] /  normalization_params["norm_velocity"]
    subhalo_vmax = subhalos["SubhaloVmax"][:] / normalization_params["norm_velocity"]
    subhalo_flag = subhalos["SubhaloFlag"][:] if hydro else np.ones_like(subhalo_halomass) # note dummy values of 1 if DMO
    halo_id = subhalos["SubhaloGrNr"][:].astype(int)

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:].astype(int)
    group_pos = halos["GroupPos"][:] / (h*1e3)
    group_vel = halos["GroupVel"][:]  / normalization_params["norm_velocity"]

    halos = pd.DataFrame(
        np.column_stack((np.arange(len(halo_mass)), group_pos, group_vel, halo_mass, halo_primarysubhalo)),
        columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mass', 'halo_primarysubhalo']
    )
    halos['halo_id'] = halos['halo_id'].astype(int)
    halos.set_index("halo_id", inplace=True)
    
    # get subhalos/galaxies      
    subhalos = pd.DataFrame(
        np.column_stack([halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_halomass, subhalo_stellarhalfmassradius, subhalo_vmax]), 
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax'],
    )
    subhalos["is_central"] = (halos.loc[subhalos.halo_id]["halo_primarysubhalo"].values == subhalos["subhalo_id"].values)
    
    # DMO-hydro matching
    if hydro:
        dmo_hydro_match = h5py.File(f"{ROOT}/illustris_data/TNG300-1/postprocessing/subhalo_matching_to_dark.hdf5")
        subhalos["subhalo_l_halo_tree"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_LHaloTree"]
        subhalos["subhalo_sublink"] = dmo_hydro_match["Snapshot_99"]["SubhaloIndexDark_SubLink"]      

    # only drop in hydro
    if hydro:
        subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
    subhalos['halo_id'] = subhalos['halo_id'].astype(int)
    subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

    subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"] / h)+10
    subhalos["subhalo_loghalomass"] = np.log10(subhalos["subhalo_halomass"] / h)+10
    subhalos["subhalo_logvmax"] = np.log10(subhalos["subhalo_vmax"])
    subhalos["subhalo_logstellarhalfmassradius"] = np.log10(subhalos["subhalo_stellarhalfmassradius"])
        
    if hydro:
        # DM cuts -- do not put outside `if` statement, otherwise indices will be missing during the `prepare_subhalos()` call
        subhalos.drop("subhalo_flag", axis=1, inplace=True)
        subhalos = subhalos[subhalos["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()
        # stellar mass and particle cuts
        # subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
        # subhalos = subhalos[subhalos["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()
    
    
    return subhalos

def prepare_subhalos(cuts=cuts):
    """Helper function to load subhalos, join to DMO simulation, add cosmic web
    parameters, and impose cuts.

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