# @Author: Arthur Voronin <arthur>
# @Date:   09.05.2021
# @Filename: test_cluster.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

import shutil
import os
import pytest
import pathlib
from unittest.mock import patch
import numpy as np
import pyrexMD.decoy.cluster as clu
import pyrexMD.misc as misc


# find main directory of pyrexMD
posixpath = pathlib.Path(".").rglob("*core.py")   # generator for matching paths
pathname = posixpath.send(None).as_posix()        # get first path name
main_dir = os.path.relpath(os.path.realpath(pathname).rstrip("core.py"))  # main directory of pyrexMD

# set up test paths
cwd = os.getcwd()
print(f"cwd: {cwd}")
pre = f"{main_dir}"
pre2 = f"{main_dir}/examples/files/cluster"
pre3 = f"{main_dir}/examples/files/rex"


def test_get_decoy_list():
    decoy_dir = f"{pre3}/decoys"
    decoy_list = clu.get_decoy_list(decoy_dir)
    expected = [f'{pre3}/decoys/1LMB_16.pdb',
                f'{pre3}/decoys/1LMB_23.pdb',
                f'{pre3}/decoys/1LMB_45.pdb',
                f'{pre3}/decoys/1LMB_96.pdb',
                f'{pre3}/decoys/1LMB_104.pdb',
                f'{pre3}/decoys/1LMB_155.pdb',
                f'{pre3}/decoys/1LMB_285.pdb',
                f'{pre3}/decoys/1LMB_303.pdb',
                f'{pre3}/decoys/1LMB_348.pdb',
                f'{pre3}/decoys/1LMB_361.pdb']
    assert np.all(decoy_list == expected)

    # coverage
    decoy_list = clu.get_decoy_list(decoy_dir+"/", ndx_range=(0, 50))
    expected = [f'{pre3}/decoys/1LMB_16.pdb',
                f'{pre3}/decoys/1LMB_23.pdb',
                f'{pre3}/decoys/1LMB_45.pdb']
    assert np.all(decoy_list == expected)
    with pytest.raises(TypeError):
        clu.get_decoy_list(None)

    return


def test_rank_cluster_decoys():
    decoy_dir = f"{pre3}/decoys"
    decoy_list = clu.get_decoy_list(decoy_dir)
    scores = misc.read_file(f"{pre3}/decoys/decoy_scores.log", usecols=1)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    BEST_DECOYS, BEST_SCORES, CLUSTER_DECOYS, CLUSTER_SCORES = clu.rank_cluster_decoys(decoy_list=decoy_list, scores=scores, labels=labels)
    expected_BEST_DECOYS = [f'{pre3}/decoys/1LMB_23.pdb', f'{pre3}/decoys/1LMB_303.pdb']
    expected_BEST_SCORES = [-240.863, -230.116]
    expected_CLUSTER_DECOYS = [[f'{pre3}/decoys/1LMB_23.pdb',
                                f'{pre3}/decoys/1LMB_104.pdb',
                                f'{pre3}/decoys/1LMB_16.pdb',
                                f'{pre3}/decoys/1LMB_96.pdb',
                                f'{pre3}/decoys/1LMB_45.pdb'],
                               [f'{pre3}/decoys/1LMB_303.pdb',
                                f'{pre3}/decoys/1LMB_348.pdb',
                                f'{pre3}/decoys/1LMB_155.pdb',
                                f'{pre3}/decoys/1LMB_285.pdb',
                                f'{pre3}/decoys/1LMB_361.pdb']]
    expected_CLUSTER_SCORES = [[-240.863, -232.453, -230.46, -227.485, -220.469],
                               [-230.116, -229.57, -229.107, -225.998, -217.567]]

    assert BEST_DECOYS == expected_BEST_DECOYS
    assert BEST_SCORES == expected_BEST_SCORES
    assert CLUSTER_DECOYS == expected_CLUSTER_DECOYS
    assert CLUSTER_SCORES == expected_CLUSTER_SCORES

    # coverage
    BEST_DECOYS, BEST_SCORES, CLUSTER_DECOYS, CLUSTER_SCORES = clu.rank_cluster_decoys(decoy_list=decoy_list, scores=scores, labels=labels, return_path=False)
    expected_CLUSTER_DECOYS = [['1LMB_23.pdb',
                                '1LMB_104.pdb',
                                '1LMB_16.pdb',
                                '1LMB_96.pdb',
                                '1LMB_45.pdb'],
                               ['1LMB_303.pdb',
                                '1LMB_348.pdb',
                                '1LMB_155.pdb',
                                '1LMB_285.pdb',
                                '1LMB_361.pdb']]
    assert CLUSTER_DECOYS == expected_CLUSTER_DECOYS
    return


def test_copy_cluster_decoys():
    decoy_dir = f"{pre3}/decoys"
    decoy_list = clu.get_decoy_list(decoy_dir)
    assert len(decoy_list) == 10
    target_dir = "./copied_decoys"

    val = clu.copy_cluster_decoys(decoy_list=decoy_list, target_dir=target_dir)
    assert val == os.path.realpath(target_dir)

    decoy_list2 = clu.get_decoy_list(os.path.realpath(target_dir))
    print("length:", len(decoy_list2))
    print("length:", len(decoy_list))
    for i in range(len(decoy_list)):
        assert misc.get_filename(decoy_list[i]) == misc.get_filename(decoy_list2[i])

    # coverage
    with pytest.raises(TypeError):
        clu.copy_cluster_decoys(decoy_list=None, target_dir=target_dir)
    with pytest.raises(TypeError):
        clu.copy_cluster_decoys(decoy_list=decoy_list, target_dir=None)
    with pytest.raises(TypeError):
        clu.copy_cluster_decoys(decoy_list=list(range(10)), target_dir=target_dir)

    # cleanup
    shutil.rmtree("./copied_decoys")
    return


def test_log_cluster_decoys():
    decoy_dir = f"{pre3}/decoys"
    decoy_list = clu.get_decoy_list(decoy_dir)
    scores = misc.read_file(f"{pre3}/decoys/decoy_scores.log", usecols=1)
    labels = np.array(range(10))

    BEST_DECOYS, BEST_SCORES, CLUSTER_DECOYS, CLUSTER_SCORES = clu.rank_cluster_decoys(decoy_list=decoy_list, scores=scores, labels=labels)

    logfile = clu.log_cluster_decoys(best_decoys=BEST_DECOYS, best_scores=BEST_SCORES, save_as="./temp.log")

    with open(f"{pre3}/decoys/decoy_scores_ranked.log", "r") as expected, open(logfile, "r") as val:
        lines1 = expected.readlines()
        lines2 = val.readlines()
        assert lines1 == lines2
    misc.rm("./temp.log")
    return


def test_save_h5():
    DM = clu.read_h5(f"{pre2}/DM")                   # coverage: append .h5
    h5_file = clu.save_h5(DM, save_as="./temp")   # coverage: append .h5

    assert h5_file == os.path.realpath("./temp.h5")
    misc.rm("./temp.h5")
    return


def test_read_h5():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)

    assert DM.shape == (500, 84, 84)
    assert np.all(np.diag(DM[0]) == np.zeros(84))
    return


def test_reshape_data():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)

    assert DM.shape == (500, 84, 84)
    DM2 = clu.reshape_data(DM, dim_out=2)
    assert DM2.shape == (500, 84*84)
    DM3 = clu.reshape_data(DM2, dim_out=3)
    assert DM3.shape == (500, 84, 84)
    return


def test_heat_KMeans():
    h5_file = f"{pre2}/DM.h5"
    cluster10 = clu.heat_KMeans(h5_file, n_clusters=10, center_type='centroid')
    assert isinstance(cluster10, clu.CLUSTER_DATA)
    assert hasattr(cluster10, "centers")
    assert hasattr(cluster10, "counts")
    assert hasattr(cluster10, "labels")
    assert hasattr(cluster10, "wss_data")
    assert hasattr(cluster10, "compact_score")
    # coverage
    clu.heat_KMeans(h5_file, n_clusters=10, center_type='medoid', stop=10)
    with pytest.raises(TypeError):
        clu.heat_KMeans(["wrong_dtype"], n_clusters=10, center_type='centroid', stop=10)
    with pytest.raises(ValueError):
        clu.heat_KMeans(h5_file, n_clusters=10, center_type='wrong_center_type', stop=10)

    return


def test_heat_KMeans_bestofN():
    h5_file = f"{pre2}/DM.h5"

    TOPX_CLUSTER = clu.heat_KMeans_bestofN(h5_file, n_clusters=10, N=5, topx=1)
    cluster_data = TOPX_CLUSTER[0]
    assert len(TOPX_CLUSTER) == 1
    assert isinstance(cluster_data, clu.CLUSTER_DATA)
    return


def test_get_DM_centroids():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    labels = [0]*500
    labels[0] == 1

    CENTROIDS = clu.get_DM_centroids(DM, labels=labels)
    print(CENTROIDS)

    # coverage
    clu.get_DM_centroids(h5_file, labels=labels)
    return


def test_get_DM_WSS():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)

    DM2 = clu.reshape_data(DM, dim_out=2)     # coverage
    h5_file2 = clu.save_h5(DM2, "./temp.h5")  # coverage
    centers = [np.zeros((84, 84))]            # coverage
    labels = [0]*len(DM2)                     # coverage
    WSS_DATA = clu.get_DM_WSS(DM2, centers=centers, labels=labels, rescale=False)
    assert isinstance(WSS_DATA, clu.WSS_DATA_obj)
    assert hasattr(WSS_DATA, "wss")
    assert hasattr(WSS_DATA, "sse")
    assert hasattr(WSS_DATA, "se_mean")
    assert hasattr(WSS_DATA, "se_std")
    misc.rm("./temp.h5")
    return


def test_get_WSS():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    TSNE = clu.apply_TSNE(DM, n_components=2)
    KMEANS = clu.apply_KMEANS(TSNE, n_clusters=5)

    WSS_DATA = clu.get_WSS(TSNE, labels=KMEANS.labels, centers=KMEANS.centers)
    assert isinstance(WSS_DATA, clu.WSS_DATA_obj)
    assert hasattr(WSS_DATA, "wss")
    assert hasattr(WSS_DATA, "sse")
    assert hasattr(WSS_DATA, "se_mean")
    assert hasattr(WSS_DATA, "se_std")

    # coverage
    WSS_DATA = clu.get_WSS(TSNE, labels=KMEANS.labels, centers=KMEANS.centers, rescale=True)
    return


@ patch("matplotlib.pyplot.show")
def test_apply_elbow_method(mock_show):
    h5_file = f"{pre2}/DM.h5"
    n_clusters = list(range(1, 10, 2))

    N_CLUSTERS, WSS = clu.apply_elbow_method(h5_file, n_clusters=n_clusters, stop=10)
    assert N_CLUSTERS == n_clusters
    for item in WSS:
        assert isinstance(item, float)

    # coverage
    with pytest.raises(TypeError):
        clu.apply_elbow_method(h5_file, n_clusters=10, stop=10)   # wrong dtype of n_clusters
    return


@ patch("matplotlib.pyplot.show")
def test_scatterplot_clustermapping(mock_show):
    xdata = list(range(10))
    ydata = list(range(10))
    labels = list(range(10))

    fig, ax = clu.scatterplot_clustermapping(xdata=xdata, ydata=ydata, labels=labels)

    # coverage
    clu.scatterplot_clustermapping(xdata=xdata, ydata=ydata, labels=labels,
                                   xlabel="xlabel", ylabel="ylabel",
                                   legend_loc="lower left", plot_only=[0, 1, 2],
                                   markers=["^" for i in range(10)])
    with pytest.raises(ValueError):
        clu.scatterplot_clustermapping(xdata=xdata, ydata=ydata, labels=[0])

    return


def test_map_cluster_scores():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    TSNE = clu.apply_TSNE(DM, n_components=2)
    KMEANS = clu.apply_KMEANS(TSNE, n_clusters=10)
    score_file = f"{pre2}/energies.log"

    scores_data = clu.map_cluster_scores(cluster_data=KMEANS, score_file=score_file)
    assert isinstance(scores_data, clu.CLUSTER_DATA_SCORES)
    assert hasattr(scores_data, "scores")
    assert hasattr(scores_data, "mean_all")
    assert hasattr(scores_data, "mean_all_filtered")
    assert hasattr(scores_data, "mean")
    assert hasattr(scores_data, "std")
    assert hasattr(scores_data, "min")
    assert hasattr(scores_data, "max")

    # coverage
    clu.map_cluster_scores(cluster_data=KMEANS, score_file=score_file, filter=False)
    return


def test_map_cluster_accuracy():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    TSNE = clu.apply_TSNE(DM, n_components=2)
    KMEANS = clu.apply_KMEANS(TSNE, n_clusters=10)

    GDT = misc.pickle_load(f"{pre2}/GDT_TS.pickle")
    RMSD = misc.pickle_load(f"{pre2}/RMSD.pickle")

    accuracy_data = clu.map_cluster_accuracy(cluster_data=KMEANS, GDT=GDT, RMSD=RMSD)
    assert isinstance(accuracy_data, clu.CLUSTER_DATA_ACCURACY)
    assert hasattr(accuracy_data, "GDT")
    assert hasattr(accuracy_data, "GDT_mean")
    assert hasattr(accuracy_data, "GDT_std")
    assert hasattr(accuracy_data, "GDT_minmax")
    assert hasattr(accuracy_data, "RMSD")
    assert hasattr(accuracy_data, "RMSD_mean")
    assert hasattr(accuracy_data, "RMSD_std")
    assert hasattr(accuracy_data, "RMSD_minmax")
    return


@ patch("matplotlib.pyplot.show")
def test_plot_cluster_data(mock_show):
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    TSNE = clu.apply_TSNE(DM, n_components=2)
    KMEANS = clu.apply_KMEANS(TSNE, n_clusters=10)

    fig, ax = clu.plot_cluster_data(cluster_data=KMEANS, tsne_data=TSNE)
    return


@ patch("matplotlib.pyplot.show")
def test_plot_cluster_centers(mock_show):
    centers = np.array([[0, 0], [1, 1]])
    cluster_data = clu.CLUSTER_DATA(centers=centers, labels=[0, 1])

    clu.plot_cluster_centers(cluster_data)
    return


def test_get_cluster_targets():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    DM = DM[:50]  # reduce n_frames for test

    TSNE = clu.apply_TSNE(DM, n_components=2)
    cluster10 = clu.apply_KMEANS(TSNE, n_clusters=10)
    cluster30 = clu.apply_KMEANS(TSNE, n_clusters=30)
    score_file = f"{pre2}/energies.log"

    n10_targets, n30_targets, n30_dist = clu.get_cluster_targets(cluster10, cluster30, score_file=score_file)
    return


def test_WF_print_cluster_accuracy_AND_test_WF_print_cluster_scores():
    h5_file = f"{pre2}/DM.h5"
    DM = clu.read_h5(h5_file)
    TSNE = clu.apply_TSNE(DM, n_components=2)
    KMEANS = clu.apply_KMEANS(TSNE, n_clusters=10)

    GDT = misc.pickle_load(f"{pre2}/GDT_TS.pickle")
    RMSD = misc.pickle_load(f"{pre2}/RMSD.pickle")
    cluster10_accuracy = clu.map_cluster_accuracy(cluster_data=KMEANS, GDT=GDT, RMSD=RMSD)

    score_file = f"{pre2}/energies.log"
    cluster10_scores = clu.map_cluster_scores(cluster_data=KMEANS, score_file=score_file)

    log1 = clu.WF_print_cluster_accuracy(cluster_data=KMEANS, cluster_accuracy=cluster10_accuracy, targets=[0], save_as="./log1.log")
    log2 = clu.WF_print_cluster_scores(cluster_data=KMEANS, cluster_scores=cluster10_scores, targets=[0], save_as="./log2.log")

    assert log1 == os.path.realpath("./log1.log")
    assert log2 == os.path.realpath("./log2.log")
    misc.rm("./log1.log")
    misc.rm("./log2.log")

    # coverage type checks of WF_print_cluster_accuracy()
    with pytest.raises(TypeError):
        clu.WF_print_cluster_accuracy(cluster_data=0, cluster_accuracy=cluster10_accuracy)
    with pytest.raises(TypeError):
        clu.WF_print_cluster_accuracy(cluster_data=KMEANS, cluster_accuracy=0)
    with pytest.raises(TypeError):
        clu.WF_print_cluster_accuracy(cluster_data=KMEANS, cluster_accuracy=cluster10_accuracy, targets=0)

    # coverage type checks of WF_print_cluster_scores()
    with pytest.raises(TypeError):
        clu.WF_print_cluster_scores(cluster_data=0, cluster_scores=cluster10_scores)
    with pytest.raises(TypeError):
        clu.WF_print_cluster_scores(cluster_data=KMEANS, cluster_scores=0)
    with pytest.raises(TypeError):
        clu.WF_print_cluster_scores(cluster_data=KMEANS, cluster_scores=cluster10_scores, targets=0)
    return
