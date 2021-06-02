# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: cluster.py
# @Last modified by:   arthur
# @Last modified time: 02.06.2021

"""
This module contains functions for:
    - decoy clustering
    - post-REX clustering (for analyses)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heat as ht
import h5py
import pyrexMD.misc as _misc
import pyrexMD.analysis.analysis as _ana
from pyrexMD.analysis.analysis import get_Distance_Matrices, _HELP_sss_None2int  # required for internal conversion
#from pyrexMD.decoy.abinitio import get_decoy_list, get_decoy_scores, get_decoy_RMSD
from tqdm.notebook import tqdm


def rank_cluster_decoys(decoy_list, scores, labels, reverse=True, return_path=True):
    """
    Rank cluster decoys based on rosetta scores.

    Args:
        decoy_list (list): output of abinitio.get_decoy_list()
        scores (list): output of abinitio.get_decoy_scores()
        labels (array, list): output of heat_KMeans()
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
        realpath (str)
            realpath of log file
    """
    if not isinstance(best_decoys, list):
        _misc.dtypeError("best_decoys", "list")
    if not isinstance(best_scores, list):
        _misc.dtypeError("best_scores", "list")

    NAMES = [_misc.get_filename(item) for item in best_decoys]
    realpath = _misc.save_table(save_as=save_as, data=[NAMES, best_scores], header="decoy name      score", verbose=verbose)
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
        prec (None, int): rounding precission of wss_data
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to True.

    Returns:
        centers (array)
            cluster centers
        counts (array)
            counts per cluster
        labels (array)
            data point cluster labels
        wss_data (tuple)
          | (WSS, SSE, SSE_mean) ~ output of get_WSS()
          | WSS (float) ~ Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
          | SSE (list) ~ Sum of Squared Errors (of individual clusters)
          | SSE_mean (list) ~ Mean values of SSE
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
        raise _misc.Error("wrong datatype: <h5_file> must be str (path to h5 file containing data).")
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
    labels = kmeans.labels_.numpy().flatten()
    counts = np.bincount(labels.flatten())
    centers = kmeans.cluster_centers_.numpy().reshape((n_clusters, int(np.sqrt(size)), int(np.sqrt(size))))
    wss_data = get_WSS(h5_file, centers=centers, counts=counts, labels=labels, verbose=False, **cfg)
    #wss, sse, sse_mean = wss_data

    if verbose:
        _misc.timeit(timer, msg="clustering time:")  # stop timer
        _misc.cprint(f"Sum of Squared Errors: {sse}")
    return centers, counts, labels, wss_data


def heat_KMeans_bestofN(h5_file, n_clusters, N=50, topx=5, verbose=True, **kwargs):
    """
    Repeat heat_KMeans N times and return topx results based on minimized sum of squared errors.


    Args:
        h5_file (str): path to h5 file containing data
        n_clusters (int): number of clusters
        N (int): number of repeats
        topx (int): number of best ranked clusters to return
        verbose (bool)

    .. Hint:: Args and Keyword Args of heat_Kmeans are valid Keyword Args.

    Returns:
        TOPX_CLUSTER (list)
            | list of clusters ranked from best to topx best. Each cluster contains:
            |   centers (array)
            |     cluster centers of best result
            |   counts (array)
            |     counts per cluster of best result
            |   labels (array)
            |     data point cluster labels
            |   wss_data (tuple)
            |     sum of squared errors
    """
    CLUSTER = []
    WSS = 99999999999999999999999999999999

    for i in tqdm(range(N)):
        cluster = heat_KMeans(h5_file, n_clusters=n_clusters, verbose=False)
        centers, counts, labels, wss_data = cluster
        # (wss, sse, sse_mean) = wss_data
        if wss_data[0] < WSS:
            CLUSTER.append(cluster)
            WSS = wss_data[0]

    TOPX_CLUSTER = CLUSTER[-topx:][::-1]
    if verbose:
        _misc.cprint(f"Returned clusters:", "blue")
        for ndx, item in enumerate(TOPX_CLUSTER):
            _misc.cprint(f"index: {ndx}\tSSE: {item[3]}", "blue")
    return TOPX_CLUSTER


def get_WSS(h5_file, centers, counts, labels, sss=[None, None, None], **kwargs):
    """
    get WSS (Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters)

    Args:
        h5file (str): path to h5_file containing distance matrices
        centers (array): output of heat_KMeans()
        counts (array): output of heat_KMeans()
        labels (array): output of heat_KMeans()
        sss (list):
          | [start, stop, step] indices of <h5_file> data
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size

    Keyword Args:
        prec (None, int): rounding precission
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to True.

    Returns:
        WSS (float)
            Within Cluster Sums of Squares ~ Sum of Squared Errors of all clusters
        SSE (list)
            Sum of Squared Errors (of individual clusters)
        SSE_mean (list)
            Mean values of SSE
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "prec": 3,
               "rescale": True}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    DM = reshape_data(read_h5(h5_file), dim_out=3, sss=[cfg.start, cfg.stop, cfg.step], verbose=False)
    SE = [[] for i in range(len(centers))]   # Squared Errors (of individual clusters)

    for ndx, l in enumerate(labels):
        d = np.linalg.norm(centers[l]-DM[ndx])
        SE[l].append(d*d)

    if cfg.rescale:
        norm = 1.0/len(DM)
    else:
        norm = 1

    # SSE: Sum of Squared Errors (of individual clusters)
    SSE = [round(norm*sum(item), cfg.prec) for item in SE]
    SSE_mean = [round(norm*np.mean(item), cfg.prec) for item in SE]

    # WSS: Within Cluster Sums of Squares == Sum of Squared Errors (of all clusters)
    WSS = round(sum(SSE), cfg.prec)
    return WSS, SSE, SSE_mean


def apply_elbow_method(h5_file, n_clusters=range(10, 31, 5), sss=[None, None, None],
                       plot=True, verbose=True, **kwargs):
    """
    Apply elbow method for a list with cluster numbers n.

    Args:
        h5file (str): path to h5_file containing distance matrices
        n_clusters (range, list, array): list with cluster numbers n to test using elbow method
        sss (list):
          | [start, stop, step] indices of <h5_file> data
          | start (None, int): start index
          | stop (None, int): stop index
          | step (None, int): step size
        plot (bool)
        verbose (bool)

    Keyword Args:
        prec (None, int): rounding precission
        rescale (bool):
          | rescale wss_data by diving through length of DM ~ n_frames.
          | Defaults to True.

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
        centers, counts, labels, wss_data = heat_KMeans(h5_file, n_clusters=i, sss=[cfg.start, cfg.step, cfg.stop],
                                                        prec=cfg.prec, rescale=cfg.rescale, verbose=False)
        # wss, sse, sse_mean = wss_data
        WSS.append(wss_data[0])

        if verbose:
            if first_print:
                _misc.cprint(f"\nN Clusters\tWSS ", "blue")
                first_print = False
            _misc.cprint(f"{i}\t{wss_data[0]}")

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
        figsize (tuple): (7,7)
        cmap (seaborn.color_palette): colormap. Defaults to seaborn.color_palette()
        marker (None, str):
          | None: use settings from Keyword Arg `markers` (see below)
          | str: ignore settings from Keyword Arg `markers` and use this marker for all scatter points
        markers (list):
          | list with markers to use for cluster mapping.
          | Rotates between markers each 10 clusters.
          | Defaults to ["o", "^", "v", "s", "P", "X"]
        alpha (float): 1
        ms (int): 6
        show_legend (bool)
        legend_loc (str): legend location. Defaults to "best".
        legend_ncols (int): legend number of columns. Defaults to 2.

    .. Hint:: Args and Keyword Args of misc.figure() are also valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    if len(xdata) != len(ydata):
        raise ValueError(f"xdata, ydata and labels must have same first dimension, but have shapes {np.shape(xdata)}, {np.shape(ydata)} and {np.shape(labels)}")
    if plot_only is None:
        plot_only = list(range(min(labels), max(labels)+1))
    default = {"figsize": (7, 7),
               "cmap": sns.color_palette(),
               "markers": ["o", "^", "v", "s", "P", "X"],
               "marker": None,
               "alpha": 1,
               "ms": 6,
               "show_legend": True,
               "legend_loc": "best",
               "legend_ncols": 2,
               "xlabel": None,
               "ylabel": None
               }
    cfg = _misc.CONFIG(default, **kwargs)
    ##################################################################
    fig, ax = _misc.figure(**cfg)
    for i in range(len(xdata)):
        if labels[i] not in plot_only:
            continue
        color = cfg.cmap[labels[i] % len(cfg.cmap)]
        if cfg.marker is None:
            marker = cfg.markers[labels[i]//len(cfg.cmap)]
        else:
            marker = cfg.marker
        plt.plot(xdata[i], ydata[i], color=color, alpha=cfg.alpha, marker=marker, ms=cfg.ms, lw=0, label=f"Cluster {labels[i]}")

    # set xlabel, ylabel
    if cfg.xlabel is not None:
        plt.xlabel(cfg.xlabel, fontweight="bold")
    if cfg.ylabel is not None:
        plt.ylabel(cfg.ylabel, fontweight="bold")

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
        ax.legend(HANDLES, LABELS, loc=cfg.legend_loc, ncol=cfg.legend_ncols)
    plt.tight_layout()
    return fig, ax


def map_cluster_scores(cluster_data, score_file, **kwargs):
    """
    map cluster scores/energies between `score_file` and `cluster_data`.

    ..Note:: `cluster_data` and `score_file` must have beeen generated using the same order of trajectory frames.
              Otherwise the index mapping from enumerate(labels) within this function will be wrong.

    Args:
        cluster_data (): output of heat_KMeans(), i.e. (centers, counts, labels, sse) = heat_KMeans()
        score_file (str): path to file containing cluster scores/energies. Logfile of abinitio.frame2score().

    Keyword Args:
        usecols (int): column of `score_file` containing scores/energies. Defaults to 1.
        skiprows (None, int): skip rows of `score_file`. Defaults to 0.
        prec (None, float): rounding precissions. Defaults to 3.

    Returns:
        CLUSTER_E (list)
            contains lists of cluster scores/energies, one for each cluster `label` in `cluster_data`
        CLUSTER_E4 (list)
            contains tuples (Emean, Estd, Emin, Emax), one for each cluster `label` in `cluster_data`
    """
    default = {"usecols": 1,
               "skiprows": 0,
               "prec": 3}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    centers, counts, labels, sse = cluster_data
    scores = _misc.read_file(score_file, usecols=cfg.usecols, skiprows=cfg.skiprows)

    CLUSTER_E = [[] for i in range(len(centers))]  # contains list of all energies, one for each cluster
    CLUSTER_E4 = []   # contains tuples (Emean, Estd, Emin, Emax), one for each cluster

    for ndx, i in enumerate(labels):
        CLUSTER_E[i].append(round(scores[ndx], cfg.prec))
    for ndx in range(len(centers)):
        Emean = round(np.mean(CLUSTER_E[ndx]), cfg.prec)
        Estd = round(np.std(CLUSTER_E[ndx]), cfg.prec)
        Emin = round(min(CLUSTER_E[ndx]), cfg.prec)
        Emax = round(max(CLUSTER_E[ndx]), cfg.prec)

        CLUSTER_E4.append((Emean, Estd, Emin, Emax))

    return CLUSTER_E, CLUSTER_E4
