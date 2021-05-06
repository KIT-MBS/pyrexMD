#from __future__ import division, print_function
import numpy as np
import heat as ht
import h5py
import pyREX.misc as _misc
from pyREX.analysis import get_Distance_Matrices, _HELP_sss_None2int  # required for internal conversion
from pyREX.abinitio import get_decoy_list, get_decoy_scores, get_decoy_RMSD


def save_h5(data, save_as, save_dir="./", HDF_group="/distance_matrices", verbose=True):
    """
    Save data (e.g. distance matrices DM) as h5 file.

    Note:
        HDF: Hierarchical Data Format
        h5: Hierarchical Data Format 5

    Args:
        data (np.array): array of distance matrices
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
        handle[HDF_group] = data
    if verbose:
        print(f"Saved h5 file as: {h5_file}")
    return h5_file


def read_h5(h5_file, HDF_group="/distance_matrices"):
    """
    read data (e.g. distance matrices DM) from h5 file.

    Note:
        HDF: Hierarchical Data Format
        h5: Hierarchical Data Format 5

    Args:
        h5_file (str)
        HDF_group (str): Hierarchical Data Format group

    Returns:
        data (np.array): data of h5 file
    """
    if _misc.get_extension(h5_file) != ".h5":
        h5_file += ".h5"
    with h5py.File(h5_file, "r") as handle:
        data = np.array(handle[HDF_group])
        return data


def reshape_data(data, dim_out=2, sss=[None, None, None], verbose=True, **kwargs):
    """
    Reshape data between the shapes: (length, size) <-> (length, sizeD1, sideD2)

    Args:
        data (np.array/ht.DNDarray)
        dim_out (int): output dimension of data
            2: output data with shape: (length, size)
            3: output data with shape: (length, sizeD1, sideD2)
        sss (list): [start, stop, step]
            start (None/int): start index
            stop (None/int): stop index
            step (None/int): step size
        verbose (bool): print messages ('reshaping data: ...')

    Kwargs:
        aliases for sss items:
            start (None/int): start index
            stop (None/int): stop index
            step (None/int): step size

    Returns:
        data (np.array/ht.DNDarray): data (same data-type as input)
    """
    ############################################################################
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]
               }
    cfg = _misc.CONFIG(default, **kwargs)
    cfg = _HELP_sss_None2int(data, cfg)  # convert None values of they keys sss, start, stop, step to integers
    data = data[cfg.start:cfg.stop:cfg.step]
    ############################################################################
    if (len(np.shape(data)) == dim_out):
        return data
    else:
        if len(np.shape(data)) == 2:
            length, size = np.shape(data)
            new_size = int(np.sqrt(size))
            data = data.reshape((length, new_size, new_size))
            if verbose:
                print(f"reshaping data: ({length}, {size}) -> ({length}, {new_size}, {new_size})")

        elif len(np.shape(data)) == 3:
            length, sizeD1, sizeD2 = np.shape(data)
            new_size = sizeD1*sizeD2
            data = data.reshape((length, new_size))
            if verbose:
                print(f"reshaping data: ({length}, {sizeD1}, {sizeD2}) -> ({length}, {new_size})")
    return data


def heat_KMeans(h5_file, HDF_group="/distance_matrices", n_clusters=20, center_type='medoid',
                sss=[None, None, None], verbose=True, **kwargs):
    """
    Use heat's KMeans Clustering

    Args:
        h5_file (str): path to h5 file containing data
        HDF_group (str): Hierarchical Data Format group
        n_clusters (int): number of clusters
        center_type (str):
            'centroid': use heat.cluster.KMeans() with centroids as cluster centers
            'medoid': use heat.cluster.KMedoids() with medoids as cluster centers
        verbose (bool)
        sss (list): [start, stop, step] indices of <h5_file> data
            start (None/int): start index
            stop (None/int): stop index
            step (None/int): step size

    Kwargs:
        aliases for sss items:
            start (None/int): start index
            stop (None/int): stop index
            step (None/int): step size
        dtype (dtype): heat.float64 (default), heat.float32, etc.

    Returns:
        centers (np.array): cluster centers
        counts (np.array): counts per cluster
        labels (np.array): data point cluster labels
    """
    default = {"dtype": ht.float64,
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)

    if isinstance(h5_file, str):
        if verbose:
            print("loading data...")
        data = ht.load(h5_file, HDF_group, split=0, dtype=cfg.dtype)
    else:
        raise _misc.Error("wrong datatype: <h5_file> must be str (path to h5 file containing data).")
    if np.shape(data) != 2:
        data = reshape_data(data, dim_out=2, sss=[cfg.start, cfg.stop, cfg.step], verbose=True)
    if verbose:
        print("clustering data...")
        timer = _misc.TIMER()
        _misc.timeit(timer)  # start timer

    length, size = np.shape(data)
    if center_type.lower() == "centroid":
        kmeans = ht.cluster.KMeans(n_clusters=n_clusters)
    elif center_type.lower() == "medoid":
        kmeans = ht.cluster.KMedoids(n_clusters=n_clusters)
    else:
        raise ValueError("""center_type must be either 'centroid' or 'medoid'.""")

    kmeans.fit(data)
    labels = kmeans.labels_.numpy()
    counts = np.bincount(labels.flatten())
    centers = kmeans.cluster_centers_.numpy().reshape((n_clusters, int(np.sqrt(size)), int(np.sqrt(size))))

    if verbose:
        _misc.timeit(timer, msg="clustering time:")  # stop timer
    return centers, counts, labels


def rank_cluster_decoys(decoy_list, scores, labels, reverse=True, return_path=True):
    """
    Rank cluster decoys based on rosetta scores.

    Args:
        decoy_list (list): output of get_decoy_list()
        scores (list): output of get_decoy_scores()
        labels (array/list): output of heat_KMeans()
        reverse (bool):
            True:  ascending ranking order (low to high)
            False: decending ranking order (high to low)
        return_path (bool):
            True:  BEST_DECOYS and CLUSTER_DECOYS contain decoy paths
            False: BEST_DECOYS and CLUSTER_DECOYS contain decoy filenames

    Returns:
        BEST_DECOYS (list): best ranked decoys (only one per cluster)
        BEST_SCORES (list): best ranked scores (only one per cluster)
        CLUSTER_DECOYS (list):
            CLUSTER_DECOYS[k]: ranked decoys of cluster k
        CLUSTER_SCORES (list):
            CLUSTER_SCORES[k]: ranked scores of cluster k
    """
    n_labels = np.max(labels) + 1

    ### link decoys and scores
    CLUSTER_DECOYS = [[] for i in range(n_labels)]
    CLUSTER_SCORES = [[] for i in range(n_labels)]
    for i in range(len(decoy_list)):
        k = labels[i].item()
        CLUSTER_DECOYS[k].append(decoy_list[i])
        CLUSTER_SCORES[k].append(scores[i])

    # rank by score
    for k in range(len(CLUSTER_DECOYS)):
        CLUSTER_DECOYS[k] = [CLUSTER_DECOYS[k][i] for i in _misc.get_ranked_array(CLUSTER_SCORES[k], reverse=reverse, verbose=False)[1]]
        CLUSTER_SCORES[k] = [CLUSTER_SCORES[k][i] for i in _misc.get_ranked_array(CLUSTER_SCORES[k], reverse=reverse, verbose=False)[1]]

    ### best per cluster
    BEST_DECOYS = [CLUSTER_DECOYS[k][0] for k in range(len(CLUSTER_DECOYS))]
    BEST_SCORES = [CLUSTER_SCORES[k][0] for k in range(len(CLUSTER_SCORES))]

    # rank by score
    for k in range(len(BEST_DECOYS)):
        BEST_DECOYS = [BEST_DECOYS[i] for i in _misc.get_ranked_array(BEST_SCORES, reverse=reverse, verbose=False)[1]]
        BEST_SCORES = [BEST_SCORES[i] for i in _misc.get_ranked_array(BEST_SCORES, reverse=reverse, verbose=False)[1]]

    if not return_path:
        BEST_DECOYS = [_misc.get_filename(item) for item in BEST_DECOYS]
        CLUSTER_DECOYS = [[_misc.get_filename(item) for item in itemList] for itemList in CLUSTER_DECOYS]

    return BEST_DECOYS, BEST_SCORES, CLUSTER_DECOYS, CLUSTER_SCORES


def copy_cluster_decoys(decoy_list, target_dir, create_dir=True, verbose=True, **kwargs):
    """
    Copy cluster decoys specified in <decoy_list> to <target_dir>.

    Args:
        decoy_list (list): output of abinitio.get_decoy_list()
                                  or cluster.get_decoy_list()
                                  or cluster.rank_cluster_decoys(return_path=True)
        target_dir (str): target directory
        create_dir (bool)
        verbose (bool)

    Kwargs:
        cprint_color (str): colored print color

    Returns:
        target_dir (str): realpath of target directory
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if not isinstance(decoy_list, list):
        raise _misc.dtypeError("decoy_list", "list")
    for item in decoy_list:
        if not isinstance(item, str):
            raise TypeError("<decoy_list> must be list of str (source paths)")
    if not (isinstance(target_dir, str)):
        raise _misc.dtypeError("target_dir", "str")

    # copy decoys if no Error occurred
    target_dir = _misc.cp(source=decoy_list, target=target_dir,
                          create_dir=create_dir, verbose=False)
    if verbose:
        _misc.cprint(f"Copied decoys to: {target_dir}", cfg.cprint_color)

    return target_dir


def log_cluster_decoys(best_decoys, best_scores, save_as, verbose=True):
    """
    Create logfile with <best_decoys> and <best_scores>.

    Args:
        best_decoys (list): output of cluster.rank_cluster_decoys()
        best_scores (list): output of cluster.rank_cluster_decoys()
        save_as (str)
        verbose (bool)

    Returns:
        realpath (str): realpath of log file
    """
    if not isinstance(best_decoys, list):
        _misc.dtypeError("best_decoys", "list")
    if not isinstance(best_scores, list):
        _misc.dtypeError("best_scores", "list")

    NAMES = [_misc.get_filename(item) for item in best_decoys]
    realpath = _misc.save_table(save_as=save_as, data=[NAMES, best_scores], header="decoy name      score", verbose=verbose)
    return realpath
