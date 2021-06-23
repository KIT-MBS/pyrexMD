# @Author: Arthur Voronin <arthur>
# @Date:   09.05.2021
# @Filename: test_gdt.py
# @Last modified by:   arthur
# @Last modified time: 23.06.2021

import pyrexMD.misc as misc
import pyrexMD.analysis.gdt as gdt
from pyrexMD.analysis.analysis import get_Distance_Matrices
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch
import pytest
import os


cwd = misc.cwd(verbose=False)

# cwd is <...>/pyrexMD/tests/
if misc.realpath(f"{cwd}").split("/")[-1] == "tests":
    pre = "./files/1l2y/"

# cwd is <...>/pyrexMD/
if misc.realpath(f"{cwd}").split("/")[-1] == "pyrexMD":
    pre = "./tests/files/1l2y/"

pdb = pre + "1l2y_ref.pdb"
tpr = pre + "traj.tpr"
xtc = pre + "traj.xtc"


def test_get_array_percent():
    p, ndx = gdt.get_array_percent(np.load(pre + "DIST_ARR.npy"), cutoff=1.0)
    v1, v2 = np.load(pre + "array_percent_1.npy", allow_pickle=True)
    assert assert_allclose(p, v1) == None
    assert assert_allclose(ndx, v2) == None

    p, ndx = gdt.get_array_percent(np.load(pre + "DIST_ARR.npy"), cutoff=2.0)
    v1, v2 = np.load(pre + "array_percent_2.npy", allow_pickle=True)
    assert assert_allclose(p, v1) == None
    assert assert_allclose(ndx, v2) == None

    p, ndx = gdt.get_array_percent(np.load(pre + "DIST_ARR.npy"), cutoff=4.0)
    v1, v2 = np.load(pre + "array_percent_4.npy", allow_pickle=True)
    assert assert_allclose(p, v1) == None
    assert assert_allclose(ndx, v2) == None
    return


def test_get_Pair_Distances():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    PAIR_DISTANCES, RMSD, resids_mobile, resids_ref = gdt.get_Pair_Distances(mobile, ref)

    assert assert_allclose(PAIR_DISTANCES, np.load(pre + "PAIR_DISTANCES.npy")) == None
    assert assert_allclose(RMSD, np.load(pre + "RMSD.npy")) == None
    assert assert_allclose(resids_mobile, np.load(pre + "resids_mobile.npy")) == None
    assert assert_allclose(resids_ref, np.load(pre + "resids_ref.npy")) == None
    return


def test_HELP_sss_None2int():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    cfg = misc.CONFIG({"sss": [None, None, None],
                       "start": None,
                       "stop": None,
                       "step": None})
    expected = misc.CONFIG({"sss": [0, 21, 1],
                            "start": 0,
                            "stop": 21,
                            "step": 1})

    val = gdt._HELP_sss_None2int(mobile, cfg)
    assert val.items() == expected.items()
    ############################################################################
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    DM = get_Distance_Matrices(mobile, stop=10)
    cfg = misc.CONFIG({"sss": [None, None, None],
                       "start": None,
                       "stop": None,
                       "step": None})
    expected = misc.CONFIG({"sss": [0, 10, 1],
                            "start": 0,
                            "stop": 10,
                            "step": 1})
    val = gdt._HELP_sss_None2int(DM, cfg)
    assert val.items() == expected.items()
    return


def test_GDT():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = gdt.GDT(mobile, ref)

    assert assert_allclose(GDT_percent, np.load(pre + "GDT_percent.npy")) == None
    assert assert_allclose(GDT_cutoff, np.load(pre + "GDT_cutoff.npy")) == None
    assert assert_allclose(RMSD, np.load(pre + "GDT_RMSD.npy")) == None
    assert assert_allclose(FRAME, np.load(pre + "GDT_FRAME.npy")) == None

    # GDT_resids has complicated structure with changing data types
    flat1 = misc.flatten_array(GDT_resids)
    flat2 = misc.flatten_array(np.load(pre + "GDT_resids.npy", allow_pickle=True))
    for i in range(len(flat1)):
        assert (flat1[i] == flat2[i]).all()
    return


def test_GDT_rna():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = gdt.GDT_rna(mobile, ref, sel1="protein and name CA", sel2="protein and name CA", cutoff=[0.5, 10, 0.5])

    assert assert_allclose(GDT_percent, np.load(pre + "GDT_percent.npy")) == None
    assert assert_allclose(GDT_cutoff, np.load(pre + "GDT_cutoff.npy")) == None
    assert assert_allclose(RMSD, np.load(pre + "GDT_RMSD.npy")) == None
    assert assert_allclose(FRAME, np.load(pre + "GDT_FRAME.npy")) == None

    # GDT_resids has complicated structure with changing data types
    flat1 = misc.flatten_array(GDT_resids)
    flat2 = misc.flatten_array(np.load(pre + "GDT_resids.npy", allow_pickle=True))
    for i in range(len(flat1)):
        assert (flat1[i] == flat2[i]).all()
    return


def test_GDT_match_cutoff_dim():
    GDT_cutoff = [1, 2]                   # random test data
    GDT_percent = [[100, 80], [80, 40]]   # random test data
    expected = [[[100, 80], [80, 40]], [[100, 80], [80, 40]]]
    val = gdt.GDT_match_cutoff_dim(GDT_cutoff, GDT_percent)
    assert val == expected


@patch("matplotlib.pyplot.show")
def test_plot_GDT(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = gdt.GDT(mobile, ref)
    fig, ax = gdt.plot_GDT(GDT_percent, GDT_cutoff)
    assert fig != None
    assert ax != None
    return


def test_get_GDT_TS():
    GDT_percent = np.load(pre + "GDT_percent.npy")
    GDT_TS = gdt.get_GDT_TS(GDT_percent)
    assert assert_allclose(GDT_TS, np.load(pre + "GDT_TS.npy")) == None
    return


def test_get_GDT_HA():
    GDT_percent = np.load(pre + "GDT_percent.npy")
    GDT_HA = gdt.get_GDT_HA(GDT_percent)
    assert assert_allclose(GDT_HA, np.load(pre + "GDT_HA.npy")) == None
    return


def test_rank_scores():
    GDT_TS = np.load(pre + "GDT_TS.npy")
    GDT_HA = np.load(pre + "GDT_HA.npy")
    rank_scores = gdt.rank_scores(GDT_TS, GDT_HA)
    assert (rank_scores == np.load(pre + "rank_scores.npy")).all()
    assert (rank_scores == np.load(pre + "GDT_rank_scores.npy")).all()

    # coverage
    gdt.rank_scores(GDT_TS, GDT_HA, ranking_order="GDT_HA")
    gdt.rank_scores(GDT_TS, GDT_HA, ranking_order=None)
    with pytest.raises(ValueError) as e_info:
        gdt.rank_scores(GDT_TS, GDT_HA, ranking_order="invalid_value")
    return


def test_GDT_rank_scores():
    GDT_percent = np.load(pre + "GDT_percent.npy")
    GDT_rank_scores = gdt.GDT_rank_scores(GDT_percent)
    assert (GDT_rank_scores == np.load(pre + "GDT_rank_scores.npy")).all()
    assert (GDT_rank_scores == np.load(pre + "rank_scores.npy")).all()
    return


def test_GDT_rank_percent():
    GDT_percent = np.load(pre + "GDT_percent.npy")
    GDT_rank_percent = gdt.GDT_rank_percent(GDT_percent)
    assert assert_allclose(GDT_rank_percent, np.load(pre + "GDT_rank_percent.npy")) == None

    # coverage
    gdt.GDT_rank_percent(GDT_percent, norm=False)
    return


def test_get_continous_segments():
    continous_segments = gdt.get_continuous_segments([1, 2, 3, 22, 23, 50, 51, 52])
    assert (continous_segments == [[1, 2, 3], [22, 23], [50, 51, 52]])
    return


def test_GDT_continous_segments():
    GDT_resids = np.load(pre + "GDT_resids.npy", allow_pickle=True)
    GDT_continous_segments = gdt.GDT_continuous_segments(GDT_resids)
    assert (GDT_continous_segments == np.load(pre + "GDT_continous_segments.npy", allow_pickle=True)).all()
    return


@ patch("matplotlib.pyplot.show")
def test_plot_LA(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    GDT = gdt.GDT(mobile, ref)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT
    SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", verbose=False)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES

    fig, ax, LA_data = gdt.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2])
    assert fig != None
    assert ax != None
    assert LA_data != None

    # coverage
    gdt.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2], show_frames=True, show_scores=True, cbar_min=0, cbar_max=2)
    gdt.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2], show_frames=True, show_scores=False, show_cbar=False, save_as="./temp.png")
    misc.rm("./temp.png")
    return


@ patch("matplotlib.pyplot.show")
def test_plot_LA_rna(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    GDT = gdt.GDT(mobile, ref)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT
    SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", verbose=False)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES

    fig, ax, LA_data = gdt.plot_LA_rna(mobile, ref, SCORES[0], SCORES[1], SCORES[2],
                                       sel1="protein and name CA",
                                       sel2="protein and name CA",
                                       cmap="GDT_TS")
    assert fig != None
    assert ax != None
    assert LA_data != None
    return
