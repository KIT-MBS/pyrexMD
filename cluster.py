from __future__ import division, print_function
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
import numpy as np
import MDAnalysis as mda
import heat as ht
import h5py
import distruct as ds
import Bio
import myPKG.misc as _misc
from myPKG.misc import HiddenPrints
from myPKG.analysis import _HELP_sss_None2int  # required for internal conversion


def get_Distance_Matrices(mobile, sss=[None, None, None], sel="protein and name CA", **kwargs):
    """
    Calculate distance matrices for mobile and return them.

    Args:
        mobile (MDA universe/atomgrp): structure with trajectory
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        sel (str): selection string
    Kwargs:
        aliases for sss items:
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size

    Returns:
        DM (np.array): array of distance matrices
    """
    ############################################################################
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]
               }
    cfg = _misc.CONFIG(default, **kwargs)
    cfg = _HELP_sss_None2int(mobile, cfg)  # convert None values of they keys sss, start, stop, step to integers
    ############################################################################
    a = mobile.select_atoms(sel)
    DM = np.empty((len(mobile.trajectory[cfg.start:cfg.stop:cfg.step]),
                   a.n_atoms*a.n_atoms))  # Args: length, size (of flattened array)

    for i, ts in enumerate(tqdm(mobile.trajectory[cfg.start:cfg.stop:cfg.step])):
        DM[i] = mda.analysis.distances.distance_array(a.positions, a.positions).flatten()
    return DM


def save_h5(data, save_as, save_dir="./", HDF_group="/distance_matrices", verbose=True):
    """
    Save data (e.g. distance matrices DM) as h5 file.

    Note:
        HDF: Hierarchical Data Format
        h5: Hierarchical Data Format 5

    Args:
        DM (np.array): array of distance matrices
        save_as (str)
        save_dir (str): save directory
            special case: save_dir is ignored when save_as is relative/absolute path
        HDF_group (str): Hierarchical Data Format group

    Returns:
        h5_file (str): realpath of h5 file
    """
    if _misc.get_extension(save_as) != ".h5":
        save_as += ".h5"

    h5_file = _misc.joinpath(save_dir, save_as)
    with h5py.File(h5_file, "w") as handle:
        handle["/distance_matrices"] = data
    if verbose:
        print(f"Saved h5 file as: {h5_file}")
    return h5_file


def read_h5(fin, HDF_group="/distance_matrices"):
    """
    read data (e.g. distance matrices DM) from h5 file.

    Note:
        HDF: Hierarchical Data Format
        h5: Hierarchical Data Format 5

    Args:
        fin (str)
        HDF_group (str): Hierarchical Data Format group

    Returns:
        data (np.array): data of h5 file
    """
    with h5py.File(fin, "r") as handle:
        data = list(handle[HDF_group])
        return data


def heat_KMeans(h5_file, HDF_group="/distance_matrices", n_clusters=20):
    """
    Use heat's KMeans Clustering

    Args:
        h5_file (str): path to h5 file (data set)
        HDF_group (str): Hierarchical Data Format group
        n_clusters (int): number of clusters

    Returns:
        kmeans (heat.cluster.kmeans.KMeans): KMeans class fitted to data set
        labels (np.array): data point cluster labels
        counts (np.array): counts per cluster
        centroids (np.array): cluster centroids
    """
    timer = _misc.TIMER()
    _misc.timeit(timer)  # start timer

    data = ht.load(h5_file, HDF_group, split=0)
    data_length, data_dim = np.shape(data)

    kmeans = ht.cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    labels = kmeans.labels_.numpy()
    counts = np.bincount(labels.flatten())
    centroids = kmeans.cluster_centers_.numpy().reshape((n_clusters, int(np.sqrt(data_dim)), int(np.sqrt(data_dim))))
    # medoids = <unsupported by heat>

    _misc.timeit(timer)  # stop timer

    return kmeans, labels, counts, centroids


def _distruct_generate_dist_file(u, DM, DM_ndx,
                                 save_as="temp.dist", save_dir="./",
                                 sel="protein and name CA", verbose=False):
    """
    Generate dist file based on protein and given distance matrix.
    (e.g. use protein and centroid -> dist file)

    Args:
        u (MDA universe/atomgrp): structure (only used to get chainids, resids etc.)
        DM (str/np.array):
            (str): path to h5 file with distance matrices
            (np.array): distance matrices
        DM_ndx: distance matrices index ~ frame which should be used
        save_as (str)
        save_dir (str): save directory
            special case: save_dir is ignored when save_as is relative/absolute path
        sel (str): selection string (limit to CA atoms only ~ distruct)
        verbose (bool)

    Returns:
        dist_file (str): realpath of dist file
    """
    default = {"HDF_group": "/distance_matrices"}
    cfg = _misc.CONFIG(default)
    ############################################################################
    a = u.select_atoms("protein and name CA")

    if _misc.get_extension(save_as) != ".dist":
        dist_file = _misc.joinpath(save_dir, f"{save_as}.dist")
    else:
        dist_file = _misc.joinpath(save_dir, save_as)

    # read distance matrix
    if isinstance(DM, str):
        DM = read_h5(DM, cfg.HDF_group)

    if len(np.shape(DM)) == 2:
        # DM are distance matrices for whole trajectory
        DM_length, DM_dim_squared = np.shape(DM)
        protein_length = int(np.sqrt(DM_dim_squared))
    elif len(np.shape(DM)) == 3:
        # DM are centroids
        DM_length, protein_length, protein_lenght = np.shape(DM)

    # write dist file for a single frame of DM
    with open(dist_file, "w") as fout:
        fix_chain_msg = True
        for i in range(len(a.names)):
            for j in range(len(a.names)):
                if i == j:
                    continue  # distruct: dist file should not contain "self edges"

                # c: chainid, r: resid, a: atomid == CA
                c1, r1, a1 = a.segids[i], a.resids[i], a.names[i]
                c2, r2, a2 = a.segids[j], a.resids[j], a.names[j]

                # fix chainid if MDA stored it as "SYSTEM" or some other weird string
                if verbose is True and fix_chain_msg is True and len(c1) != 1:
                    print(f"Fixing chain id for dist file (distruct): {c1} -> A")
                    fix_chain_msg = False
                if len(c1) != 1:
                    c1 = "A"
                if len(c2) != 1:
                    c2 = "A"

                weight = 1.0
                if DM_length == 1:
                    distance = DM.reshape(protein_length, protein_length)[i][j]
                else:
                    distance = DM[DM_ndx].reshape(protein_length, protein_length)[i][j]

                fout.write(f"{c1} {r1} {a1} {c2} {r2} {a2} {distance} {weight}\n")
        if verbose:
            print(f"Saved dist file as: {dist_file}")
    return dist_file


def distruct_generate_structure(u, DM, DM_ndx, pdbid, seq,
                                save_as="default", save_dir="./structures",
                                verbose=True, verbose_distruct=False, **kwargs):
    """
    Use distruct to generate structure.

    Args:
        u (MDA universe/atomgrp): structure (only used to get chainids, resids etc.)
        DM (str/np.array):
            (str): path to h5 file with distance matrices
            (np.array): distance matrices
        DM_ndx: distance matrices index ~ frame which should be used
        pdbid (str)
        seq (str): fasta sequence
        save_as (str): structure file saved as .pdb or .cif
            "default": ./structures/<pdbid>_<DM_ndx>.pdb
        save_dir (str): save directory
            special case: save_dir is ignored when save_as is relative/absolute path
        verbose (bool)
        verbose_distruct (bool): show/hide distruct prints

    Kwargs:
        save_format (str): "pdb", "cif"

    Returns:
        io_file (str): realpath of .pdb/.cif file (generated structure)
    """
    default = {"save_as": save_as,
               "save_dir": save_dir,
               "save_format": "pdb"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # read sequence
    seqs = [Bio.Seq.Seq(seq, Bio.Alphabet.generic_protein)]

    # write temporary dist file
    dist_file = _distruct_generate_dist_file(u, DM, DM_ndx, verbose=False)

    # read list of contacts
    contacts = []
    with open(dist_file, 'r') as f:
        for line in f:
            # c: chainid, r: resid, a: atomid == CA
            (c1, r1, a1, c2, r2, a2, distance, weight) = line.split()
            c = (((c1, int(r1), a1), (c2, int(r2), a2)), float(distance), float(weight))
            contacts.append(c)
        f.close()

    # generate structure
    if verbose_distruct:
        s = ds.Distructure(pdbid, seqs)  # "identifier/name", list of sequences (in case of protein complex)
        s.generate_primary_contacts()
        s.set_tertiary_contacts(contacts)
        s.run()
    else:
        with HiddenPrints():
            s = ds.Distructure(pdbid, seqs)  # "identifier/name", list of sequences (in case of protein complex)
            s.generate_primary_contacts()
            s.set_tertiary_contacts(contacts)
            s.run()

    # save structure as .pdb or .cif
    if save_as == "default":
        io_file = _misc.joinpath(cfg.save_dir, f"/{pdbid}_{DM_ndx}.{cfg.save_format}", create_dir=True)
    elif ".pdb" not in _misc.get_extension(save_as) or ".cif" not in _misc.get_extension(save_as):
        io_file = _misc.joinpath(cfg.save_dir, f"{cfg.save_as}.{cfg.save_format}", create_dir=True)

    if cfg.save_format == "pdb":
        io = Bio.PDB.PDBIO()
    elif cfg.save_format == "cif":
        io = Bio.PDB.mmcifio.MMCIFIO()
    io.set_structure(s)
    io.save(io_file)
    if verbose:
        print(f"Generated structure: {io_file}")

    # delete temporary dist file
    _misc.rm(dist_file, verbose=False)
    return io_file
