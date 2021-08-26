# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: cluster.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

"""
.. hint:: This module contains functions for:

    - decoy clustering
    - post-REX clustering (for analyses)

Example:
--------

    import pyrexMD.misc as misc
    import pyrexMD.analysis.cluster as clu

    # load data
    QDATA = misc.pickle_load("./data/QDATA.pickle")
    RMSD = misc.pickle_load("./data/RMSD.pickle")
    GDT_TS = misc.pickle_load("./data/GDT_TS.pickle")
    score_file = "./data/energies.log"
    ENERGY = misc.read_file(score_file, usecols=1, skiprows=1)
    DM = clu.read_h5("./data/DM.h5")

    # apply TSNE for dimension reduction
    tsne = clu.apply_TSNE(DM, n_components=2, perplexity=50, random_state=1)

    ### apply KMeans on TSNE-transformed data (two variants with low and high cluster number)
    # note: here we set the high number only to 20 because our sample is small with only 500 frames
    cluster10 = clu.apply_KMEANS(tsne, n_clusters=10, random_state=1)
    cluster20 = clu.apply_KMEANS(tsne, n_clusters=20, random_state=1)

    ### map scores (energies) and accuracy (GDT, RMSD) to clusters
    cluster10_scores = clu.map_cluster_scores(cluster_data=cluster10, score_file=score_file)
    cluster10_accuracy = clu.map_cluster_accuracy(cluster_data=cluster10, GDT=GDT_TS, RMSD=RMSD)
    cluster20_scores = clu.map_cluster_scores(cluster_data=cluster20, score_file=score_file)
    cluster20_accuracy = clu.map_cluster_accuracy(cluster_data=cluster20, GDT=GDT_TS, RMSD=RMSD)

    ### plot cluster data
    # here: TSNE-transformed data with n_clusters = 10
    # also: plot cluster centers with different colors
    #     - red dot: n20 centers
    #     - black dot: n10 centers
    clu.plot_cluster_data(cluster10, tsne, ms=40)
    clu.plot_cluster_center(cluster10, marker="o", color="red", ms=20)
    clu.plot_cluster_center(cluster20, marker="o", color="black")

    ### plot cluster data
    # here: TSNE-transformed data with n_clusters = 20
    # also: plot cluster centers with different colors
    #     - red dot: n20 centers
    #     - black dot: n10 centers
    clu.plot_cluster_data(cluster20, tsne)
    clu.plot_cluster_center(cluster10, marker="o", color="red", ms=20)
    clu.plot_cluster_center(cluster20, marker="o", color="black")

    ### print table with cluster scores stats
    clu.WF_print_cluster_scores(cluster_data=cluster10, cluster_scores=cluster10_scores)
    clu.WF_print_cluster_scores(cluster_data=cluster20, cluster_scores=cluster20_scores)

    ### print table with cluster accuracy stats
    clu.WF_print_cluster_accuracy(cluster_data=cluster10, cluster_accuracy=cluster10_accuracy)
    clu.WF_print_cluster_accuracy(cluster_data=cluster20, cluster_accuracy=cluster20_accuracy)

Content:
--------
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heat as ht
import h5py
import pyrexMD.misc as _misc
import pyrexMD.analysis.analysis as _ana
from pyrexMD.analysis.analysis import get_Distance_Matrices, _HELP_sss_None2int  # required for internal conversion
#from pyrexMD.analysis.abinitio import get_decoy_list, get_decoy_scores, get_decoy_RMSD
import pandas as pd
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from tqdm.notebook import tqdm
import glob
import os


def get_decoy_list(decoy_dir, pattern="*.pdb", ndx_range=(None, None)):
    """
    | Alias function of get_structure_list().
    | get decoy list(sorted by a numeric part at any position of the filename,
    | e.g. 1LMB_1.pdb, 1LMB_2.pdb, ...)

    Args:
        decoy_dir (str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple, list): limit decoy index range to [ndx_min, ndx_max]

    Returns:
        DECOY_LIST (list)
            list with decoy filenames
    """
    if not isinstance(decoy_dir, str):
        raise TypeError("type(decoy_dir) must be string.")
    if decoy_dir[-1] == "/":
        decoy_dir = decoy_dir[:-1]

    # preset stuff
    decoy_list = glob.glob(f"{decoy_dir}/{pattern}")
    decoy_names = [_misc.get_filename(i) for i in decoy_list]
    decoy_names_substrings = [_misc.get_substrings(i) for i in decoy_names]

    # get decoy index range
    diff_element = list(set(decoy_names_substrings[0])-set(decoy_names_substrings[1]))
    if len(diff_element) != 1:
        raise ValueError('decoy_names_substrings differ in more than 1 element (i.e. differ at more than 1 position)')
    diff_position = decoy_names_substrings[0].index(diff_element[0])
    min_ndx = 999999999
    max_ndx = -999999999
    for i in decoy_names_substrings:
        if i[diff_position].isdigit() and int(i[diff_position]) < min_ndx:
            min_ndx = int(i[diff_position])
        if i[diff_position].isdigit() and int(i[diff_position]) > max_ndx:
            max_ndx = int(i[diff_position])

    if ndx_range[0] is not None:
        min_ndx = ndx_range[0]
    if ndx_range[1] is not None:
        max_ndx = ndx_range[1]

    # get decoy filename prefix and suffix
    template = decoy_names[0]
    template_diff_element = _misc.get_substrings(template)[diff_position]
    decoy_filename_prefix = template[:template.index(template_diff_element)]
    decoy_filename_suffix = template[template.index(template_diff_element)+len(template_diff_element):]

    # create sorted decoy list
    DECOY_LIST = [f"{decoy_dir}/{decoy_filename_prefix}{i}{decoy_filename_suffix}" for i in range(min_ndx, max_ndx+1)
                  if f"{decoy_dir}/{decoy_filename_prefix}{i}{decoy_filename_suffix}" in decoy_list]
    return DECOY_LIST


def rank_cluster_decoys(decoy_list, scores, labels, reverse=True, return_path=True):
    """
    Rank cluster decoys based on rosetta scores.

    Args:
        decoy_list (list): output of abinitio.get_decoy_list()
        scores (list): output of abinitio.get_decoy_scores()
        labels (array, list): output of heat_KMeans() or apply_KMeans()
        reverse (bool):
          | True:  ascending ranking order (low to high)
          | False: decending ranking order (high to low)
        return_path (bool):
          | True:  BEST_DECOYS and CLUSTER_DECOYS contain decoy paths
          | False: BEST_DECOYS and CLUSTER_DECOYS contain decoy filenames

    Returns:
        BEST_DECOYS (list)
            best ranked decoys (only one per cluster)
        BEST_SCORES (list)
            best ranked scores (only one per cluster)
        CLUSTER_DECOYS (list)
            CLUSTER_DECOYS[k] are ranked decoys of cluster k
        CLUSTER_SCORES (list)
            CLUSTER_SCORES[k] are ranked scores of cluster k
    """
    n_labels = np.max(labels) + 1

    ### link decoys and scores
    CLUSTER_DECOYS = [[] for i in range(n_labels)]
    CLUSTER_SCORES = [[] for i in range(n_labels)]
    for ndx in range(len(decoy_list)):
        k = labels[ndx]
        CLUSTER_DECOYS[k].append(decoy_list[ndx])
        CLUSTER_SCORES[k].append(scores[ndx])

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
        decoy_list (list):
          | output of abinitio.get_decoy_list()
          | or cluster.rank_cluster_decoys(return_path=True)
        target_dir (str): target directory
        create_dir (bool)
        verbose (bool)

    Keyword Args:
        cprint_color (str): colored print color

    Returns:
        target_dir (str)
            realpath of target directory
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
    target_dir = _misc.cp(source=decoy_list, target=target_dir, create_dir=create_dir, verbose=False)

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
        realpath (str)
            realpath of log file
    """
    if not isinstance(best_decoys, list):
        _misc.dtypeError("best_decoys", "list")
    if not isinstance(best_scores, list):
        _misc.dtypeError("best_scores", "list")

    NAMES = [_misc.get_filename(item) for item in best_decoys]
    realpath = _misc.save_table(save_as=save_as, data=[NAMES, best_scores], header="#decoy name     score", verbose=verbose)
    return realpath

################################################################################
################################################################################


def save_h5(data, save_as, save_dir="./", HDF_group="/distance_matrices", verbose=True):
    """
    Save data (e.g. distance matrices DM) as h5 file.

    .. Note::
      | HDF: Hierarchical Data Format
      | h5: Hierarchical Data Format 5

    Args:
        data (array): array of distance matrices
        save_as (str)
        save_dir (str):
          | save directory
          | special case: save_dir is ignored when save_as is relative/absolute path
        HDF_group (str): Hierarchical Data Format group

    Returns:
        h5_file (str)
            realpath of h5 file
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
    read h5 data (e.g. distance matrices DM)

    .. Note::
      | HDF: Hierarchical Data Format
      | h5: Hierarchical Data Format 5

    Args:
        h5_file (str)
        HDF_group (str): Hierarchical Data Format group

    Returns:
        data (array)
            data of h5 file
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
        data (array, heat.DNDarray)
        dim_out (int):
          | output dimension of data
          | 2: output data with shape: (length, size)
          | 3: output data with shape: (length, sizeD1, sideD2)
        sss (list):
          | [start, stop, step]
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size
        verbose (bool): print messages ('reshaping data: ...')

    Keyword Args:
        start (None, int): start index
        stop (None, int): stop index
        step (None, int): step size

    Returns:
        data (array, ht.DNDarray)
            reshaped data with same data-type as input
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


def heat_KMeans(h5_file, HDF_group="/distance_matrices", n_clusters=20, center_type='centroid',
                sss=[None, None, None], verbose=True, **kwargs):
    """
    apply heat's KMeans clustering algorithm

    Args:
        h5_file (str): path to h5 file containing data
        HDF_group (str): Hierarchical Data Format group
        n_clusters (int): number of clusters
        center_type (str):
          | 'centroid': use heat.cluster.KMeans() with centroids as cluster centers
          | 'medoid': use heat.cluster.KMedoids() with medoids as cluster centers
        verbose (bool)
        sss (list):
          | [start, stop, step] indices of <h5_file> data
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size

    Keyword Args:
        start (None, int): start index
        stop (None, int): stop index
        step (None, int): step size
        dtype (dtype): heat.float64 (default), heat.float32, etc.
        prec (None, int): rounding precision of wss_data
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to True.

    Returns:
        cluster_data (CLUSTER_DATA)
          | .centers (array)
          |     cluster centers
          | .counts (array)
          |     counts per cluster
          | .labels (array)
          |     data point cluster labels
          | .noise_label (None, int)
          |     noise label used for algorithms such as DBSCAN
          | .wss_data (WSS_DATA)
          |      output of get_WSS() or get_DM_WSS()
          |      .wss_data.wss (float)
          |         Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          |      .wss_data.sse (list)
          |         Sum of Squared Errors (of individual clusters)
          |      .wss_data.se_mean (list)
          |             mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          |      .wss_data.se_std (list)
          |             std values of Squared Errors for each cluster
          | .compact_score (array)
          |      mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
    """
    default = {"dtype": ht.float64,
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "rescale": True}
    cfg = _misc.CONFIG(default, **kwargs)

    if isinstance(h5_file, str):
        if verbose:
            print("loading data...")
        data = ht.load(h5_file, HDF_group, split=0, dtype=cfg.dtype)
    else:
        raise TypeError("wrong datatype: <h5_file> must be str (path to h5 file containing data).")
    if np.shape(data) != 2:
        data = reshape_data(data, dim_out=2, sss=[cfg.start, cfg.stop, cfg.step], verbose=verbose)
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
    centers = kmeans.cluster_centers_.numpy().reshape((n_clusters, int(np.sqrt(size)), int(np.sqrt(size))))
    counts = np.bincount(kmeans.labels_.numpy().flatten())
    labels = kmeans.labels_.numpy().flatten()

    wss_data = get_DM_WSS(h5_file, centers=centers, labels=labels, verbose=False, **cfg)
    cluster_data = CLUSTER_DATA(centers=centers, counts=counts, labels=labels, wss_data=wss_data, compact_score=wss_data.se_mean)

    if verbose:
        _misc.timeit(timer, msg="clustering time:")  # stop timer
        _misc.cprint(f"WSS: {wss_data.wss}")
    return cluster_data


def heat_KMeans_bestofN(h5_file, n_clusters, N=50, topx=5, verbose=True, **kwargs):
    """
    Repeat heat_KMeans N times and return topx results based on minimized sum of squared errors.

    Args:
        h5_file (str): path to h5 file containing data
        n_clusters (int): number of clusters
        N (int): number of repeats
        topx (int): number of best ranked clusters to return
        verbose (bool)

    .. Hint:: Args and Keyword Args of heat_KMeans are valid Keyword Args.

    Returns:
        TOPX_CLUSTER (list)
          | list of clusters ranked from best to topx best. Each cluster contains CLUSTER_DATA with:
          |     .centers (array)
          |         cluster centers of best result
          |     .counts (array)
          |         counts per cluster of best result
          |     .labels (array)
          |         data point cluster labels
          |     .wss_data (WSS_DATA)
          |         output of get_WSS() or get_DM_WSS()
          |         .wss (float)
          |             Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          |         .sse (list)
          |             Sum of Squared Errors (of individual clusters)
          |         .se_mean (list)
          |             mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          |         .se_std (list)
          |             std values of Squared Errors for each cluster
          |     .compact_score (array)
          |         mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
    """
    CLUSTER = []
    WSS = 99999999999999999999999999999999

    for i in tqdm(range(N)):
        cluster_data = heat_KMeans(h5_file, n_clusters=n_clusters, verbose=False)
        if cluster_data.wss_data.wss < WSS:
            CLUSTER.append(cluster_data)
            WSS = cluster_data.wss_data.wss

    TOPX_CLUSTER = CLUSTER[-topx:][::-1]
    if verbose:
        _misc.cprint("Returned clusters:", "blue")
        for ndx, item in enumerate(TOPX_CLUSTER):
            _misc.cprint(f"index: {ndx}\tWSS: {item.wss_data.wss}", "blue")
    return TOPX_CLUSTER


def get_DM_centroids(DM, labels):
    """
    get Distance Matrix centroids.

    Args:
        DM (str, array): path to h5_file containing distance matrices or array with distance matrices
        labels (array): cluster labels for each frame of DM

    Returns:
        CENTROIDS (array)
            centroids of DM, one for each label
    """
    if isinstance(DM, str):
        DM = read_h5(DM)
    CENTROIDS = []

    DM_map = [[] for i in range(min(labels), max(labels)+1)]
    for ndx, item in enumerate(labels):
        DM_map[item].append(DM[ndx])

    for dms in DM_map:
        CENTROIDS.append(sum(dms)/len(dms))

    return np.array(CENTROIDS)


def get_DM_WSS(DM, centers, labels, sss=[None, None, None], **kwargs):
    """
    get distance matrix WSS (Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters)

    Args:
        DM (str, array): path to h5_file containing distance matrices or array with distance matrices
        centers (array): cluster centers array with dim(DM) == dim(centers).
        labels (array): cluster labels array with length(DM)
        sss (list):
          | [start, stop, step] indices of DM data
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size

    Keyword Args:
        prec (None, int): rounding precision
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to False.

    Returns:
        WSS_DATA (WSS_DATA)
          | .wss (float)
          |     Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          | .sse (array)
          |     Sum of Squared Errors (of individual clusters)
          | .se_mean (array)
          |     mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          | .se_std (array)
          |     std values of Squared Errors for each cluster
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "prec": 3,
               "rescale": False}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if isinstance(DM, str):
        DM = reshape_data(read_h5(DM), dim_out=3, sss=[cfg.start, cfg.stop, cfg.step], verbose=False)
    else:
        DM = reshape_data(DM, dim_out=3, sss=[cfg.start, cfg.stop, cfg.step], verbose=False)

    SE = [[] for i in range(len(centers))]   # Squared Errors (of individual clusters)

    for ndx, l in enumerate(labels):
        d = np.linalg.norm(centers[l]-DM[ndx])
        SE[l].append(d*d)

    if cfg.rescale:
        norm = 1.0/len(DM)
    else:
        norm = 1

    # statistics
    SE_mean = [round(norm*np.mean(item), cfg.prec) for item in SE]
    SE_std = [round(norm*np.std(item), cfg.prec) for item in SE]

    # SSE: Sum of Squared Errors (of individual clusters)
    SSE = [round(norm*sum(item), cfg.prec) for item in SE]
    # WSS: Within Cluster Sums of Squares == Sum of Squared Errors (of all clusters)
    WSS = round(sum(SSE), cfg.prec)

    WSS_DATA = WSS_DATA_obj(wss=WSS, sse=SSE, se_mean=SE_mean, se_std=SE_std)
    return WSS_DATA


def get_WSS(data, centers, labels, **kwargs):
    """
    get WSS (Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters)

    Args:
        data (array): array with data, for example distance matrices or TSNE-transformed data.
        centers (array): cluster centers array with dim(data) == dim(centers).
        labels (array): cluster labels array with lenght of data.

    Keyword Args:
        prec (None, int): rounding precision
        rescale (bool):
          | rescale wss_data by diving through length of data ~ n_frames.
          | Defaults to False.

    Returns:
        WSS_DATA (WSS_DATA)
          | .wss (float)
          |     Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          | .sse (array)
          |     Sum of Squared Errors (of individual clusters)
          | .se_mean (array)
          |     mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          | .se_std (array)
          |     std values of Squared Errors for each cluster
    """
    default = {"prec": 3,
               "rescale": False}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    SE = [[] for i in range(len(centers))]   # Squared Errors (of individual clusters)

    for ndx, l in enumerate(labels):
        d = np.linalg.norm(centers[l]-data[ndx])
        SE[l].append(d*d)

    if cfg.rescale:
        norm = 1.0/len(data)
    else:
        norm = 1

    # statistics
    SE_mean = [round(norm*np.mean(item), cfg.prec) for item in SE]
    SE_std = [round(norm*np.std(item), cfg.prec) for item in SE]

    # SSE: Sum of Squared Errors (of individual clusters)
    SSE = [round(norm*sum(item), cfg.prec) for item in SE]
    # WSS: Within Cluster Sums of Squares == Sum of Squared Errors (of all clusters)
    WSS = round(sum(SSE), cfg.prec)

    WSS_DATA = WSS_DATA_obj(wss=WSS, sse=SSE, se_mean=SE_mean, se_std=SE_std)
    return WSS_DATA


def apply_elbow_method(h5_file, n_clusters=range(10, 31, 5), sss=[None, None, None],
                       plot=True, verbose=True, **kwargs):
    """
    Apply elbow method for a list with cluster numbers n.

    Args:
        h5_file (str): path to h5_file containing distance matrices
        n_clusters (range, list, array): list with cluster numbers n to test using elbow method
        sss (list):
          | [start, stop, step] indices of <h5_file> data
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size
        plot (bool)
        verbose (bool)

    Keyword Args:
        prec (None, int): rounding precision
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to False.

    .. Hint:: Args and Keyword Args of misc.figure() are valid Keyword Args.

    Returns:
        N_CLUSTERS (list)
            list with cluster numbers n to test
        WSS (list)
           list of wss scores ~ Within Cluster Sums of Squares

    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "prec": 3,
               "rescale": True}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if not isinstance(n_clusters, (list, range, np.ndarray)):
        raise TypeError("n_clusters must be range, list, np.ndarray")

    N_CLUSTERS = list(n_clusters)
    WSS = []    # wss of each test case
    first_print = True

    for i in tqdm(n_clusters):
        cluster_data = heat_KMeans(h5_file, n_clusters=i, sss=[cfg.start, cfg.step, cfg.stop],
                                   prec=cfg.prec, rescale=cfg.rescale, verbose=False)
        # wss, sse,  = wss_data
        WSS.append(cluster_data.wss_data.wss)

        if verbose:
            if first_print:
                _misc.cprint("\nN Clusters\tWSS ", "blue")
                first_print = False
            _misc.cprint(f"{i}\t{cluster_data.wss_data.wss}")

    if plot:
        _ana.PLOT(xdata=N_CLUSTERS, ydata=WSS, xlabel="Number of Clusters", ylabel="Sum of Squared Errors", **kwargs)
    return N_CLUSTERS, WSS


def scatterplot_clustermapping(xdata, ydata, labels, plot_only=None, **kwargs):
    """
    Creates scatterplot for (xdata, ydata) with cluster mapping according to labels.

    Args:
        xdata (list, array)
        ydata (list, array)
        labels (list, array): list with cluster labels used to map (xdata, ydata) ~ Output of heat_KMeans().
        plot_only (None, list): list of cluster labels to plot. Ignores (xdata, ydata) of other labels.

    Keyword Args:
        figsize (tuple): (6.6, 6.6)
        cmap (seaborn.color_palette): colormap. Defaults to seaborn.color_palette(n_colors=60)
        markers_list (list): Defaults to ["o", "^", "s", "v", "P", "X"]
        markers_repeats (list): Defaults to [10, 10, 10, 10, 10, 10]. Specifies how often each marker should be repeated before changing to the next marker.
        markers (list): list of markers used to plot. Will be generated based on markers_list and markers_repeats or can be passed directly.
        alpha (float): 1
        ms (int): 6
        show_legend (bool)
        legend_loc (str): legend location ("best", "upper left" etc.). Defaults to "outside".
        legend_ncols (int): legend number of columns. Defaults to 2.
        xlabel (None, str)
        ylabel (None, str)
        xlim (None, tuple): (xmin, xmax). Defaults to (None, None).
        ylim (None, tuple): (ymin, ymax). Defaults to (None, None).

    .. Hint:: Args and Keyword Args of misc.figure() are also valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    if len(xdata) != len(ydata) or len(xdata) != len(labels):
        raise ValueError(f"xdata, ydata and labels must have same first dimension, but have shapes {np.shape(xdata)}, {np.shape(ydata)} and {np.shape(labels)}")
    if plot_only is None:
        plot_only = list(range(min(labels), max(labels)+1))
    default = {"figsize": (6.6, 6.6),
               "cmap": sns.color_palette(n_colors=60),
               "markers_list": ["o", "^", "s", "v", "P", "X"],
               "markers_repeats": [10, 10, 10, 10, 10, 10],
               "alpha": 1,
               "ms": 20,
               "show_legend": True,
               "legend_loc": "outside",
               "legend_ncols": 1,
               "xlabel": None,
               "ylabel": None,
               "xlim": (None, None),
               "ylim": (None, None)}
    size = len(set(labels))
    default = _misc.CONFIG(default, **kwargs)
    default_markers = [default.markers_list[ndx] for ndx, item in enumerate(default.markers_repeats) for _ in range(item)]
    modified = {"cmap": default.cmap[:size],
                "markers": default_markers[:size]}
    cfg = _misc.CONFIG(default, **modified)   # use default settings, overwrite with modified settings
    cfg = _misc.CONFIG(cfg, **kwargs)     # overwrite with kwargs settings
    ##################################################################
    fig, ax = _misc.figure(**cfg)
    for ndx, i in enumerate(labels):
        if i not in plot_only:
            continue
        if cfg.markers[i] in ["^" or "v"]:
            plt.scatter(xdata[ndx], ydata[ndx], color=cfg.cmap[i], alpha=cfg.alpha, marker=cfg.markers[i], s=int(1.5*cfg.ms), lw=0, label=f"Cluster {i}")
        else:
            plt.scatter(xdata[ndx], ydata[ndx], color=cfg.cmap[i], alpha=cfg.alpha, marker=cfg.markers[i], s=cfg.ms, lw=0, label=f"Cluster {i}")

    # set xlabel, ylabel
    if cfg.xlabel is not None:
        plt.xlabel(cfg.xlabel, fontweight="bold")
    if cfg.ylabel is not None:
        plt.ylabel(cfg.ylabel, fontweight="bold")

    # set xlim, ylim
    if isinstance(cfg.xlim, (tuple, list)):
        plt.xlim((cfg.xlim[0], cfg.xlim[1]))
    if isinstance(cfg.ylim, (tuple, list)):
        plt.ylim((cfg.ylim[0], cfg.ylim[1]))

    # create sorted legend
    if cfg.show_legend:
        _handles, _labels = ax.get_legend_handles_labels()
        LABELS = [f"Cluster {i}" for i in plot_only]
        HANDLES = []
        for i, label in enumerate(LABELS):
            for j, item in enumerate(_labels):
                if item == label:
                    HANDLES.append(_handles[j])
                    break

        # plot legend outside
        if cfg.legend_loc == "outside":
            ax.legend(HANDLES, LABELS, ncol=cfg.legend_ncols, bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize='xx-small')
        # plot legend inside
        else:
            ax.legend(HANDLES, LABELS, loc=cfg.legend_loc, ncol=cfg.legend_ncols)

    plt.tight_layout()
    return fig, ax


def map_cluster_scores(cluster_data, score_file, filter=True, filter_tol=2.0, **kwargs):
    """
    map cluster scores/energies between `cluster_data` and `score_file`.

    ..Note:: `cluster_data` and `score_file` must have beeen generated using the same order of trajectory frames.
              Otherwise the index mapping from enumerate(labels) within this function will be wrong.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        score_file (str): path to file containing cluster scores/energies. Logfile of abinitio.frame2score()
        filter (bool): apply filter ~ use only scores s which fulfil: filter_min <= s <= filter_max
        filter_tol (float):
          | defines filter_min and filter_max values
          | filter_min = np.mean(scores) - filter_tol * np.std(scores)
          | filter_max = np.mean(scores) + filter_tol * np.std(scores)

    Keyword Args:
        usecols (int): column of `score_file` containing scores/energies. Defaults to 1.
        skiprows (None, int): skip rows of `score_file`. Defaults to 0.
        prec (None, float): rounding precisions. Defaults to 3.

    Returns:
        cluster_scores (CLUSTER_DATA_SCORES)
          | .scores (list)
          |   summary of all scores of individual clusters
          | .mean_all (float)
          |   mean value of all scores before applying filter
          | .mean_all_filtered (float)
          |   mean value of all scores after applying filter
          | .mean (list)
          |   mean values of individual cluster scores
          | .std (list)
          |   std values of individual cluster scores
          | .min (list)
          |   min values of individual cluster scores
          | .max (list)
          |    max values of individual cluster scores
    """
    default = {"usecols": 1,
               "skiprows": 0,
               "prec": 3}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    scores = _misc.read_file(score_file, usecols=cfg.usecols, skiprows=cfg.skiprows)
    cluster_scores = CLUSTER_DATA_SCORES(cluster_data=cluster_data, scores=scores, filter=filter, filter_tol=filter_tol)
    return cluster_scores


def map_cluster_accuracy(cluster_data, GDT, RMSD, prec=3):
    """
    map cluster accuracy between `cluster_data`, cluster `GDT` and cluster `RMSD`.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        GDT (list, array): GDT data for each frame of cluster_data
        RMSD (list, array): RMSD data for each frame of cluster_data
        prec (None, int): rounding precision

    Returns:
      | cluster_accuracy (CLUSTER_DATA_ACCURACY)
      |   .GDT (array)
      |       summary of all GDT values of individual clusters
      |   .GDT_mean (array)
      |       GDT mean values of individual clusters
      |   .GDT_std (array)
      |       GDT std values of individual clusters
      |   .GDT_minmax (array)
      |       [GDT_min, GDT_max] values of individual clusters
      |   .RMSD (array)
      |       summary of all RMSD values of individual clusters
      |   .RMSD_mean (array)
      |       RMSD mean values of individual clusters
      |   .RMSD_std (array)
      |       RMSD std values of individual clusters
      |   .RMSD_minmax (array)
      |       [RMSD_min, RMSD_max] values of individual clusters
    """
    cluster_accuracy = CLUSTER_DATA_ACCURACY(cluster_data, GDT=GDT, RMSD=RMSD, prec=prec)
    return cluster_accuracy


def sort_cluster_data(cluster_data, cluster_accuracy):
    """
    sort cluster data based on GDT_mean values of cluster_accuracy.
      -> cluster 0 will have highest GDT_mean
      -> cluster <max> will have lowest GDT_mean

    .. Note :: if cluster_data has noise_label assigned, will move this label to the end of the sorted cluster data.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        cluster_accuracy (CLUSTER_DATA_ACCURACY): output of map_cluster_accuracy()

    Returns:
        sorted_cluster_data (CLUSTER_DATA)
            sorted cluster_data
    """
    if not isinstance(cluster_data, CLUSTER_DATA):
        raise TypeError("cluster_data has wrong data type.")
    if not isinstance(cluster_accuracy, CLUSTER_DATA_ACCURACY):
        raise TypeError("cluster_accuracy has wrong data type.")

    ### rank and test if labels have same range
    ranked_array, ranked_ndx = _misc.get_ranked_array(cluster_accuracy.GDT_mean, verbose=False)
    if set(cluster_data.labels) != set(ranked_ndx):
        raise ValueError("labels of cluster_data and cluster_accuracy do not match.")

    if cluster_data.noise_label is not None:
        # move noise label to the very end
        noise_ndx = np.where(ranked_ndx == cluster_data.noise_label)[0]
        other_ndx = np.where(ranked_ndx != cluster_data.noise_label)[0]
        ranked_array = np.append(ranked_array[other_ndx], ranked_array[noise_ndx])
        ranked_ndx = np.append(ranked_ndx[other_ndx], ranked_ndx[noise_ndx])

    ### REMOVE ###
    # # algorithms with -1 label for noise (e.g. DBSCAN, OPTICS)
    # else:
    #     ranked_array, ranked_ndx = _misc.get_ranked_array(cluster_accuracy.GDT_mean, verbose=False)
    #     if set(cluster_data.labels + 1) != set(ranked_ndx):
    #         raise ValueError("labels of cluster_data and cluster_accuracy do not match.")
    #     # move noise label (here: max(ranked_ndx)) to the very end
    #     noise_ndx = np.where(ranked_ndx == max(ranked_ndx))[0]
    #     other_ndx = np.where(ranked_ndx != max(ranked_ndx))[0]
    #     ranked_array = np.append(ranked_array[other_ndx], ranked_array[noise_ndx])
    #     ranked_ndx = np.append(ranked_ndx[other_ndx], ranked_ndx[noise_ndx])
    ### REMOVE ###

    ### remap data
    sorted_labels = [ranked_ndx.tolist().index(i) for i in cluster_data.labels]

    ### REMOVE ###
    # # algorithms with -1 label for noise (e.g. DBSCAN, OPTICS)
    # else:
    #     _misc.cprint(f"Note: shifted labels from {min(cluster_data.labels)}..{max(cluster_data.labels)} to {min(ranked_ndx)}..{max(ranked_ndx)} with {max(ranked_ndx)} being the 'noise'.", "red")
    #     sorted_labels = [ranked_ndx.tolist().index(i) for i in cluster_data.labels + 1]  # shift labels
    ### REMOVE ###

    ### create new object
    sorted_wss_data = WSS_DATA_obj(wss=cluster_data.wss_data.wss,                      # float
                                   sse=cluster_data.wss_data.sse[ranked_ndx],          # sorted
                                   se_mean=cluster_data.wss_data.se_mean[ranked_ndx],  # sorted
                                   se_std=cluster_data.wss_data.se_std[ranked_ndx])    # sorted
    sorted_cluster_data = CLUSTER_DATA(centers=cluster_data.centers[ranked_ndx],  # sorted
                                       counts=cluster_data.counts[ranked_ndx],    # sorted
                                       labels=sorted_labels,                      # sorted
                                       noise_label=cluster_data.noise_label,      # reassign
                                       inertia=cluster_data.inertia,              # float
                                       wss_data=sorted_wss_data,                  # sorted
                                       compact_score=cluster_data.compact_score[ranked_ndx])  # sorted
    return sorted_cluster_data


def apply_MDS(data, n_components=2, random_state=None):
    """
    apply multidimensional scaling on data.

    Args:
        data (array)
        n_components (int): MDS number of components
        random_state (None, int): Determines the random number generator

    Returns:
        mds_data (array)
            MDS transformed data
    """
    nsamples, nx, ny = np.array(data).shape
    data_reshaped = np.array(data).reshape((nsamples, nx*ny))
    mds_data = MDS(n_components=n_components, random_state=random_state).fit_transform(data_reshaped)
    return mds_data


def apply_TSNE(data, n_components=2, perplexity=50, random_state=None):
    """
    apply t-distributed stochastic neighbor embedding on data.

    Args:
        data (array)
        n_components (int): TSNE number of components
        perplexity (int): TSNE perplexity
        random_state (None, int): Determines the random number generator

    Returns:
        tsne_data (array)
            tsne transformed data
    """
    nsamples, nx, ny = np.array(data).shape
    data_reshaped = np.array(data).reshape((nsamples, nx*ny))
    tsne_data = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(data_reshaped)
    return tsne_data


def apply_DBSCAN(data, eps=0.5, min_samples=20):
    """
    apply DBSCAN on data

    Args:
        data (array)
        eps (float): maximum (epsilon) distance between two samples to be considered neighbors. Most important DBSCAN parameter.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point during DBSCAN algorithm.

    Returns:
        cluster_data (CLUSTER_DATA)
          | .centers (array)
          |     cluster centers
          | .counts (array)
          |     counts per cluster
          | .labels (array)
          |     data point cluster labels
          | .noise_label (None, int)
          |     noise label used for algorithms such as DBSCAN
          | .inertia (None, float)
          |     inertia of data
          | .wss_data (WSS_DATA)
          |      output of get_WSS() or get_DM_WSS()
          |      .wss_data.wss (float)
          |         Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          |      .wss_data.sse (list)
          |         Sum of Squared Errors (of individual clusters)
          |      .wss_data.se_mean (list)
          |             mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          |      .wss_data.se_std (list)
          |             std values of Squared Errors for each cluster
          | .compact_score (array)
          |      mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    labels = np.array([item if item != -1 else max(dbscan.labels_)+1 for item in dbscan.labels_])  # DBSCAN uses -1 label for noise. remap noise to highest index.
    noise_label = max(dbscan.labels_)+1
    centers = get_cluster_centers(data=data, labels=labels)
    counts = np.bincount(labels)
    wss_data = get_WSS(data, centers=centers, labels=labels)

    cluster_data = CLUSTER_DATA(centers=centers, counts=counts, labels=labels,
                                noise_label=noise_label, inertia=None,
                                wss_data=wss_data, compact_score=wss_data.se_mean)
    return cluster_data


def apply_KMEANS(tsne_data, n_clusters=30, random_state=None):
    """
    apply KMeans on tsne_data

    Args:
        tnse_data (array): output of apply_TSNE()
        n_clusters (int): number of clusters
        random_state (None, int): Determines the random number generator(for centroid initialization)

    Returns:
        cluster_data (CLUSTER_DATA)
          | .centers (array)
          |     cluster centers
          | .counts (array)
          |     counts per cluster
          | .labels (array)
          |     data point cluster labels
          | .noise_label (None, int)
          |     noise label used for algorithms such as DBSCAN
          | .inertia (None, float)
          |     inertia of data
          | .wss_data (WSS_DATA)
          |      output of get_WSS() or get_DM_WSS()
          |      .wss_data.wss (float)
          |         Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          |      .wss_data.sse (list)
          |         Sum of Squared Errors (of individual clusters)
          |      .wss_data.se_mean (list)
          |             mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
          |      .wss_data.se_std (list)
          |             std values of Squared Errors for each cluster
          | .compact_score (array)
          |      mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=None).fit(tsne_data)

    centers = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    wss_data = get_WSS(tsne_data, centers=centers, labels=labels)

    cluster_data = CLUSTER_DATA(centers=centers, counts=counts, labels=labels, inertia=inertia, wss_data=wss_data, compact_score=wss_data.se_mean)
    return cluster_data


def plot_cluster_data(cluster_data, tsne_data, plot_only=None, **kwargs):
    """
    plot cluster data in a scatter plot.

    .. Note:: Uses modified default values if cluster_data has n_clusters = 10 or n_clusters = 30.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        tsne_data (array): output of apply_TSNE()
        plot_only (None, list): list of cluster labels to plot. Ignores (xdata, ydata) of other labels.

    Keyword Args:
        cmap (sns.color_palette(n_colors))
        markers_list (list): Defaults to ["o", "^", "s", "v", "P", "X"]
        markers_repeats (list): Defaults to [10, 10, 10, 10, 10, 10]. Specifies how often each marker should be repeated before changing to the next marker.
        markers (list): list of markers used to plot. Will be generated based on markers_list and markers_repeats or can be passed directly.
        ms (int): marker size. Defaults to 40.
        figsize (tuple): Defaults to (6.6, 5.6)
        aspect ('auto', 'equal', int):
          | aspect ratio of figure. Defaults to 'auto'.
          | 'auto': fill the position rectangle with data.
          | 'equal': synonym for aspect=1, i.e. same scaling for x and y.
          | int: a circle will be stretched such that the height is *int* times the width.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class)
            matplotlib.axes._subplots.Axes
    """
    size = len(cluster_data.centers)
    default = {"cmap": sns.color_palette(n_colors=60),
               "markers_list": ["o", "^", "s", "v", "P", "X"],
               "markers_repeats": [10, 10, 10, 10, 10, 10],
               "ms": 40,
               "figsize": (6.6, 6.6),
               "aspect": "auto"}
    default = _misc.CONFIG(default, **kwargs)
    default_markers = [default.markers_list[ndx] for ndx, item in enumerate(default.markers_repeats) for _ in range(item)]
    modified = {"cmap": default.cmap[:size],
                "markers": default_markers[:size]}
    cfg = _misc.CONFIG(default, **modified)   # use default settings, overwrite with modified settings
    cfg = _misc.CONFIG(cfg, **kwargs)     # overwrite with kwargs settings
    ###################################################################
    df = pd.DataFrame({'X': tsne_data[:, 0],
                       'Y': tsne_data[:, 1],
                       'label': cluster_data.labels})
    if plot_only is not None:
        for i in sorted(set(cluster_data.labels)):
            if i not in plot_only:
                df = df[df.label != i]
    # PLOT
    fig, ax = _misc.figure(figsize=cfg.figsize)
    sns.scatterplot(x="X", y="Y",
                    hue="label", palette=cfg.cmap,
                    style="label", markers=cfg.markers,
                    s=cfg.ms,
                    legend='full',
                    data=df)
    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")

    # plot legend
    h, l = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax.legend(h, l, ncol=2, bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize='xx-small')
    ax.set_aspect(cfg.aspect)
    plt.tight_layout()

    return fig, ax


def plot_cluster_centers(cluster_data, plot_only=None, color="black", marker="s", ms=10):
    """
    plot cluster centers in existing figure. If cluster_data has a noise_label
    assigned it will be automatically ignored and not plotted.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        plot_only (None, list):
          | list of cluster labels to plot. Ignores (xdata, ydata) of other labels.
          | None: plot all labels
          | list: plot only labels within list
        color (str)
        marker (str)
        ms (int): marker size
    """
    X = cluster_data.centers[:, 0]
    Y = cluster_data.centers[:, 1]

    if plot_only is None:
        plot_only = sorted(set(cluster_data.labels))
    if cluster_data.noise_label in plot_only:
        plot_only.remove(cluster_data.noise_label)
    for label in plot_only:
        plt.scatter(X[label], Y[label], color=color, marker=marker, s=ms)

    return


def get_cluster_centers(data, labels):
    """
    get cluster centers of data.


    Args:
        data (np.array): data with shape (n_samples, n_dim)
        labels (np.array): data labels with length = n_samples

    Returns:
        centers (np.array)
            cluster centers
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have equal length.")

    centers = []
    for label in sorted(set(labels)):
        current_data = np.array([data[ndx] for ndx, item in enumerate(labels) if item == label])
        centers.append(sum(current_data)/len(current_data))
    centers = np.array(centers)
    return centers


def get_cluster_targets(cluster_data_n10, cluster_data_n30, score_file, prec=3, verbose=True):
    """
    get cluster targets by finding optimum center in n10 and then ranking distances
    to n30 centers. This does not 'really' select targets but rather provides
    additional information to select proper cluster targets based on

        - cluster energy mean values
        - distance between cluster centers with low energy mean (low distance == similar structures)
        - compact score

    .. Note:: n30 clusters with very low mean and low compact score can be good
       targets too, even if their cluster centers are not close to n10 target.

    Args:
        cluster_data_n10 (CLUSTER_DATA): ~ output of apply_KMEANS(n_clusters=10) or heat_KMEANS(n_clusters=10)
        cluster_data_n30 (CLUSTER_DATA): ~ output of apply_KMEANS(n_clusters=30) or heat_KMEANS(n_clusters=10)
        score_file (str): ~ output of abinitio.frame2score()
        prec (None, int): rounding precision of distance

    Returns:
        n10_targets (array)
            n10 target labels (ranked by Emean of n10 clusters) ~ use only one or two n10 targets
        n30_targets (array)
            n30 target labels (ranked by distance from low to high)
        n30_dist (array)
            n30 target distances, with distance n30_dist[i] = n10_centers[n10_targets[0]] - n30_centers[i]
    """
    cluster_scores_n10 = map_cluster_scores(cluster_data=cluster_data_n10, score_file=score_file)
    #cluster_scores_n30 = map_cluster_scores(cluster_data=cluster_data_n30, score_file=score_file)

    # find optimum center of n10 (lowest mean)
    _, n10_targets = _misc.get_ranked_array(cluster_scores_n10.mean, reverse=True, verbose=False)
    n10_center = cluster_data_n10.centers[n10_targets[0]]
    n10, n30 = len(cluster_data_n10.counts), len(cluster_data_n30.counts)

    # get center distances d[i] = n10_center - n30_center[i]
    size = len(cluster_data_n30.centers)
    D = np.zeros((size))
    for i in range(size):
        D[i] = np.linalg.norm(n10_center-cluster_data_n30.centers[i])

    # rank center distances (low to high)
    n30_dist, n30_targets = _misc.get_ranked_array(D, reverse=True, verbose=False)
    n30_dist = np.around(n30_dist, prec)
    if verbose:
        _misc.cprint(f"'n{n10} target' based on min(Emean): {n10_targets[0]}", "red")
        _misc.cprint(f"'n{n10} target' distance to adjacent n{n30} clusters:", "red")
        _misc.cprint("\nndx  distance", "blue")
        # print 5 closest cluster centers
        for ndx in range(5):
            _misc.cprint(f"{n30_targets[ndx]:>3}   {n30_dist[ndx]:>6}")
    return (n10_targets, n30_targets, n30_dist)

################################################################################
################################################################################
### WF functions


def WF_print_cluster_accuracy(cluster_data, cluster_accuracy, targets=[None], save_as=None):
    """
    Workflow to print and log cluster accuracy.

    .. Note:: 'compact score' is mean(squared errors) of individual clusters.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        cluster_accuracy (CLUSTER_DATA_ACCURACY): output of map_cluster_accuracy()
        targets (list): target labels, which will be colored red in printed table
        save_as (None, str): realpath or name of logfile

    Returns:
        save_as (None, str)
            realpath of logfile
    """
    if not isinstance(cluster_data, CLUSTER_DATA):
        raise TypeError("Wrong datatype of cluster_data.")
    if not isinstance(cluster_accuracy, CLUSTER_DATA_ACCURACY):
        raise TypeError("Wrong datatype of cluster_accuracy.")
    if not isinstance(targets, (list, np.ndarray)):
        raise TypeError("type(targets) must be either list or array.")
    # rank clusters by mean GDT value
    _ranked, _ndx = _misc.get_ranked_array(cluster_accuracy.GDT_mean, verbose=False)

    # print
    _misc.cprint(f"cluster n{len(cluster_data.counts)} accuracy (ranked by GDT mean)\n", "red")
    _misc.cprint("                   | GDT     GDT    GDT     GDT    | RMSD   RMSD   RMSD   RMSD", "blue")
    _misc.cprint("ndx  size  compact | mean    std    min     max    | mean   std    min    max", "blue")
    for ndx in _ndx:
        if targets is not [None] and ndx in targets:
            cprint_color = "red"
        else:
            cprint_color = None
        _misc.cprint(f"{ndx:>3}   {cluster_data.counts[ndx]:^4} {cluster_data.compact_score[ndx]:>7} | {cluster_accuracy.GDT_mean[ndx]:<6}  {cluster_accuracy.GDT_std[ndx]:<6} {cluster_accuracy.GDT_minmax[ndx][0]:<6}  {cluster_accuracy.GDT_minmax[ndx][1]:<6} | {cluster_accuracy.RMSD_mean[ndx]:<5}  {cluster_accuracy.RMSD_std[ndx]:<5}  {cluster_accuracy.RMSD_minmax[ndx][0]:<5}  {cluster_accuracy.RMSD_minmax[ndx][1]:<5}", cprint_color)

    # save log
    if save_as is not None:
        with open(save_as, "w") as fout:
            fout.write(f"cluster n{len(cluster_data.counts)} accuracy (ranked by GDT mean)\n\n")
            fout.write("                   | GDT     GDT    GDT     GDT    | RMSD   RMSD   RMSD   RMSD\n")
            fout.write("ndx  size  compact | mean    std    min     max    | mean   std    min    max\n")
            for ndx in _ndx:
                fout.write(f"{ndx:>3}   {cluster_data.counts[ndx]:^4} {cluster_data.compact_score[ndx]:>7} | {cluster_accuracy.GDT_mean[ndx]:<6}  {cluster_accuracy.GDT_std[ndx]:<6} {cluster_accuracy.GDT_minmax[ndx][0]:<6}  {cluster_accuracy.GDT_minmax[ndx][1]:<6} | {cluster_accuracy.RMSD_mean[ndx]:<5}  {cluster_accuracy.RMSD_std[ndx]:<5}  {cluster_accuracy.RMSD_minmax[ndx][0]:<5}  {cluster_accuracy.RMSD_minmax[ndx][1]:<5}\n")
        print(f"\n Saved log as: {save_as}")
        return os.path.realpath(save_as)
    return


def WF_print_cluster_scores(cluster_data, cluster_scores, targets=[None], save_as=None):
    """
    Workflow to print and log cluster scores.

    .. Note:: 'compact score' is mean(squared errors) of individual clusters.

    Args:
        cluster_data (CLUSTER_DATA): output of apply_DBSCAN(), apply_KMeans(), or heat_KMeans()
        cluster_scores (CLUSTER_DATA_SCORES): output of map_cluster_scores()
        targets (list): target labels, which will be colored red in printed table
        save_as (None, str): realpath or name of logfile

    Returns:
        save_as (None, str)
            realpath of logfile
    """
    if not isinstance(cluster_data, CLUSTER_DATA):
        raise TypeError("Wrong datatype of cluster_data.")
    if not isinstance(cluster_scores, CLUSTER_DATA_SCORES):
        raise TypeError("Wrong datatype of cluster_scores.")
    if not isinstance(targets, (list, np.ndarray)):
        raise TypeError("type(targets) must be either list or array.")
    # rank by Emean
    _ranked, _ndx = _misc.get_ranked_array(cluster_scores.mean, reverse=True, verbose=False)

    _misc.cprint(f"cluster n{len(cluster_data.counts)} scores (ranked by Emean)", "red")
    _misc.cprint("\nndx  size  compact | Emean    Estd    Emin      Emax      DELTA", "blue")
    for ndx in _ndx:
        if targets is not [None] and ndx in targets:
            cprint_color = "red"
        else:
            cprint_color = None
        if cluster_scores.DELTA[ndx] < 0:
            _misc.cprint(f"{ndx:>3}  {cluster_data.counts[ndx]:^4}  {cluster_data.compact_score[ndx]:>7} |{cluster_scores.mean[ndx]:<8}  {cluster_scores.std[ndx]:<5}  {cluster_scores.min[ndx]:<8}  {cluster_scores.max[ndx]:<8}  {cluster_scores.DELTA[ndx]:<6}", cprint_color)
        else:
            # move scores by 1 digit if positive (looks nicer)
            _misc.cprint(f"{ndx:>3}  {cluster_data.counts[ndx]:^4}  {cluster_data.compact_score[ndx]:>7} |{cluster_scores.mean[ndx]:<8}  {cluster_scores.std[ndx]:<5}  {cluster_scores.min[ndx]:<8}  {cluster_scores.max[ndx]:<8}   {cluster_scores.DELTA[ndx]:<6}", cprint_color)

    # save log
    if save_as is not None:
        with open(save_as, "w") as fout:
            fout.write(f"cluster n{len(cluster_data.counts)} scores (ranked by Emean)\n")
            fout.write("\nndx  size  compact | Emean    Estd    Emin      Emax     DELTA\n")
            for ndx in _ndx:
                if cluster_scores.DELTA[ndx] < 0:
                    fout.write(f"{ndx:>3}  {cluster_data.counts[ndx]:^4}  {cluster_data.compact_score[ndx]:>7} |{cluster_scores.mean[ndx]:<8}  {cluster_scores.std[ndx]:<5}  {cluster_scores.min[ndx]:<8}  {cluster_scores.max[ndx]:<8}  {cluster_scores.DELTA[ndx]:<6}\n")
                else:
                    # move scores by 1 digit if positive (looks nicer)
                    fout.write(f"{ndx:>3}  {cluster_data.counts[ndx]:^4}  {cluster_data.compact_score[ndx]:>7} |{cluster_scores.mean[ndx]:<8}  {cluster_scores.std[ndx]:<5}  {cluster_scores.min[ndx]:<8}  {cluster_scores.max[ndx]:<8}   {cluster_scores.DELTA[ndx]:<6}\n")
        print(f"\n Saved log as: {save_as}")
        return os.path.realpath(save_as)
    return
################################################################################
################################################################################
### CLASSES / OBJECTS


class CLUSTER_DATA(object):
    def __init__(self, centers=None, counts=None, labels=None, noise_label=None, inertia=None, wss_data=None, compact_score=None):
        """
        saves cluster data as object

        Args:
            centers (None, array): cluster centers
            counts (None, array): counts
            labels (None, array): labels
            noise_label (None, int): noise label used for book-keeping for certain algorithms, such as DBSCAN
            inertia (None, float): intertia
            wss_data (None, WSS_DATA)
            compact_score (None, array): mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
        """
        self.centers = np.array(centers)
        self.counts = np.array(counts)
        self.labels = np.array(labels)
        self.noise_label = noise_label
        self.inertia = inertia
        self.wss_data = wss_data
        self.compact_score = np.array(compact_score)
        return


class CLUSTER_DATA_SCORES(object):
    def __init__(self, cluster_data, scores, prec=3, filter=True, filter_tol=2.0):
        """
        mapping between cluster_data and scores.

        Args:
            cluster_data (CLUSTER_DATA)
            scores (list, array): scores of frames in cluster_data
            prec (int): rounding precision
            filter (bool): apply filter ~ use only scores s which fulfil: filter_min <= s <= filter_max
            filter_tol (float):
              | defines filter_min and filter_max values
              | filter_min = np.mean(scores) - filter_tol * np.std(scores)
              | filter_max = np.mean(scores) + filter_tol * np.std(scores)

        Returns:
            obj.scores (list)
                summary of all scores of individual clusters
            obj.mean_all (float)
                mean value of all scores before applying filter
            obj.mean_all_filtered (float)
                mean value of all scores after applying filter
            obj.mean (list)
                mean values of individual cluster scores
            obj.std (list)
                std values of individual cluster scores
            obj.min (list)
                min values of individual cluster scores
            obj.max (list)
                max values of individual cluster scores
            obj.DELTA (list)
                difference between individual cluster mean and total mean, i.e. DELTA[i] = mean[i] - mean_all
        """
        self.scores = [[] for i in range(len(cluster_data.centers))]
        if filter:
            self._filter_min = round(np.mean(scores)-filter_tol*np.std(scores), prec)
            self._filter_max = round(np.mean(scores)+filter_tol*np.std(scores), prec)
        self.mean_all = round(np.mean(scores), prec)
        self.mean_all_filtered = round(np.mean(scores), prec)
        self.mean = []
        self.std = []
        self.min = []
        self.max = []
        self.DELTA = []

        # process scores
        for ndx, i in enumerate(cluster_data.labels):
            s = scores[ndx]
            if not filter:
                self.scores[i].append(round(s, prec))
            if filter and self._filter_min <= s <= self._filter_max:
                self.scores[i].append(round(s, prec))
        if filter:
            self.mean_all_filtered = round(np.mean(_misc.flatten_array(self.scores)), prec)

        # get mean, std, etc.
        for ndx in range(len(cluster_data.centers)):
            self.mean.append(round(np.mean(self.scores[ndx]), prec))
            self.std.append(round(np.std(self.scores[ndx]), prec))
            self.min.append(round(min(self.scores[ndx]), prec))
            self.max.append(round(max(self.scores[ndx]), prec))

            # DELTA[i] = mean[i] - mean_all
            self.DELTA.append(round(self.mean[ndx]-self.mean_all_filtered, prec))

        return


class CLUSTER_DATA_ACCURACY(object):
    def __init__(self, cluster_data, GDT, RMSD, prec=3):
        """
        mapping between cluster_data and accuracy (GDT and RMSD).

        Args:
            cluster_data (CLUSTER_DATA)
            GDT (list, array): GDT data for each frame of cluster_data
            RMSD (list, array): RMSD data for each frame of cluster_data
            prec (int): rounding precision

        Returns:
            obj.GDT (list)
                summary of all GDT values of individual clusters
            obj.GDT_mean (array)
                GDT mean values of individual clusters
            obj.GDT_std (array)
                GDT std values of individual clusters
            obj.GDT_minmax (array)
                [GDT_min, GDT_max] values of individual clusters
            obj.RMSD (list)
                summary of all RMSD values of individual clusters
            obj.RMSD_mean (array)
                RMSD mean values of individual clusters
            obj.RMSD_std (array)
                RMSD std values of individual clusters
            obj.RMSD_minmax (array)
                [RMSD_min, RMSD_max] values of individual clusters
        """
        n_labels = len(set(cluster_data.labels))

        self.GDT = [[] for i in range(n_labels)]
        self.GDT_mean = []
        self.GDT_std = []
        self.GDT_minmax = []
        self.RMSD = [[] for i in range(n_labels)]
        self.RMSD_mean = []
        self.RMSD_std = []
        self.RMSD_minmax = []

        # process scores
        for ndx, i in enumerate(cluster_data.labels):
            self.GDT[i].append(GDT[ndx])
            self.RMSD[i].append(RMSD[ndx])

        for ndx in range(n_labels):
            self.GDT_mean.append(np.mean(self.GDT[ndx]))
            self.GDT_std.append(np.std(self.GDT[ndx]))
            self.GDT_minmax.append([min(self.GDT[ndx]), max(self.GDT[ndx])])
            self.RMSD_mean.append(np.mean(self.RMSD[ndx]))
            self.RMSD_std.append(np.std(self.RMSD[ndx]))
            self.RMSD_minmax.append([min(self.RMSD[ndx]), max(self.RMSD[ndx])])

        if prec is not None:
            for ndx, item in enumerate(self.GDT):
                self.GDT[ndx] = np.around(item, prec)
            self.GDT_mean = np.around(self.GDT_mean, prec)
            self.GDT_std = np.around(self.GDT_std, prec)
            self.GDT_minmax = np.around(self.GDT_minmax, prec)
            for ndx, item in enumerate(self.RMSD):
                self.RMSD[ndx] = np.around(item, prec)
            self.RMSD_mean = np.around(self.RMSD_mean, prec)
            self.RMSD_std = np.around(self.RMSD_std, prec)
            self.RMSD_minmax = np.around(self.RMSD_minmax, prec)

        return


class WSS_DATA_obj(object):
    def __init__(self, wss=None, sse=None, se_mean=None, se_std=None):
        """
        | .wss (float)
        |     Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
        | .sse (array)
        |     Sum of Squared Errors (of individual clusters)
        | .se_mean (array)
        |     mean values of Squared Errors for each cluster ~ can be interpreted as a compactness score
        | .se_std (array)
        |     std values of Squared Errors for each cluster
        """
        self.wss = wss
        self.sse = np.array(sse)
        self.se_mean = np.array(se_mean)
        self.se_std = np.array(se_std)
        return
