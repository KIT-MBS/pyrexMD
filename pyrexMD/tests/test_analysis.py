# @Author: Arthur Voronin <arthur>
# @Date:   10.05.2021
# @Filename: test_analysis.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021


import pyrexMD.analysis.analysis as ana
import pyrexMD.misc as misc
import MDAnalysis as mda
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from unittest.mock import patch
import pathlib
import pytest
import os


# find main directory of pyrexMD
posixpath = pathlib.Path(".").rglob("*core.py")   # generator for matching paths
pathname = posixpath.send(None).as_posix()        # get first path name
main_dir = os.path.relpath(os.path.realpath(pathname).rstrip("core.py"))  # main directory of pyrexMD

# set up test paths
cwd = os.getcwd()
print(f"cwd: {cwd}")
pre = f"{main_dir}/tests/files/1l2y"
pdb = f"{pre}/1l2y_ref.pdb"
tpr = f"{pre}/traj.tpr"
xtc = f"{pre}/traj.xtc"


def test_get_timeconversion():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ana.get_time_conversion(mobile, tu="ns")

    mobile.trajectory.units["time"] = "ns"
    ana.get_time_conversion(mobile, tu="ps")

    with pytest.raises(TypeError):
        ana.get_time_conversion("wrong_type", tu="ps")
    return


def test_get_FASTA():
    with pytest.warns(PDBConstructionWarning):
        assert ana.get_FASTA(pdb) == ['NLYIQWLKDGGPSSGRPPPS']
    return


def test_alignto():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    val = ana.alignto(mobile, ref, "protein and name CA", "protein and name CA")
    assert assert_allclose(val, (0.37619036475899914, 0.3503198898884487)) == None
    return


@patch("matplotlib.pyplot.show")
def test_get_RMSD(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    val = ana.get_RMSD(mobile, ref, sel1='backbone', sel2='backbone', plot=True)
    expected = np.load(f"{pre}/get_RMSD.npy")
    assert assert_allclose(val[0], expected[0]) == None
    assert assert_allclose(val[1], expected[1]) == None
    assert assert_allclose(val[2], expected[2]) == None
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_RMSF(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    val = ana.get_RMSF(mobile, "backbone", plot=True)
    expected = np.load(f"{pre}/get_RMSF.npy")
    assert assert_allclose(val, expected) == None

    # coverage 2nd plot case
    val = ana.get_RMSF(mobile, "protein and name CA", plot=True)
    with pytest.raises(TypeError):
        val = ana.get_RMSF("wrong_type", "backbone", plot=True)
    plt.close("all")
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

    val = ana._HELP_sss_None2int(mobile, cfg)
    assert val.items() == expected.items()
    ############################################################################
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    DM = ana.get_Distance_Matrices(mobile, stop=10)
    cfg = misc.CONFIG({"sss": [None, None, None],
                       "start": None,
                       "stop": None,
                       "step": None})
    expected = misc.CONFIG({"sss": [0, 10, 1],
                            "start": 0,
                            "stop": 10,
                            "step": 1})
    val = ana._HELP_sss_None2int(DM, cfg)
    assert val.items() == expected.items()
    return


def test_get_Distance_Matrices():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    val = ana.get_Distance_Matrices(mobile)
    expected = np.load(f"{pre}/get_Distance_Matrices.npy")
    assert assert_allclose(val, expected) == None

    # coverage
    mobile_list = [f"{pdb}", f"{pdb}"]
    ana.get_Distance_Matrices(mobile_list, flatten=True)
    ana.get_Distance_Matrices(mobile_list, flatten=False)
    with pytest.raises(TypeError):
        mobile_list = [f"{pdb}", f"non_pdb_extension.txt"]
        ana.get_Distance_Matrices(mobile_list)
    return


def test_get_shortest_RES_distances():
    ref = mda.Universe(pdb)
    val = ana.get_shortest_RES_distances(ref, sel="protein")
    expected = np.load(f"{pre}/shortest_RES_distances.npy", allow_pickle=True)
    assert assert_allclose(val[0], expected[0]) == None
    assert (val[1] == expected[1]).all()  # this could cause problems as mixed types

    # coverage
    ana.get_shortest_RES_distances(pdb, sel="protein")          # type string
    ana.get_shortest_RES_distances(ref.atoms, sel="protein")    # type atom grp
    return


def test_get_trendline():
    X = [0, 1, 2, 3, 4, 5]
    Y = [0, 10, 20, 20, 5, 15]
    X_val, Y_val = ana.get_trendline(X, Y, compress=2)

    X_expected = [0.5, 2.5, 4.5]
    Y_expected = [5.0, 20.0, 10.0]
    assert X_val == X_expected
    assert Y_val == Y_expected

    # coverage
    ana.get_trendline(np.array(X), np.array(Y), compress=2)   # type: array
    ana.get_trendline(np.array(X), np.array(Y), compress=20)  # remainder != 0
    with pytest.raises(ValueError):
        ana.get_trendline(X, Y[:-2], compress=2)
    return


@patch("matplotlib.pyplot.show")
def test_plot_trendline(mock_show):
    X = [0, 1, 2, 3, 4, 5]
    Y = [0, 10, 20, 20, 5, 15]
    trendline = ana.plot_trendline(X, Y, compress=2)
    assert isinstance(trendline, matplotlib.lines.Line2D)

    # coverage
    fig, ax = misc.figure()
    ana.plot_trendline(X, Y, compress=2, fig=fig)  # use existing figure: fit type
    ana.plot_trendline(X, Y, compress=2, fig=fig)  # remove existing trendline
    ana.plot_trendline(X, Y, compress=2, fig=5)    # use existing figure: int type
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_remove_trendline(mock_show):
    X = [0, 1, 2, 3, 4, 5]
    Y = [0, 10, 20, 20, 5, 15]
    trendline = ana.plot_trendline(X, Y, compress=2)
    ana.remove_trendline()

    #coverage: Figure has no trendline object (or already removed)
    ana.remove_trendline()

    # coverage: trendline passed
    fig, ax = misc.figure()
    trendline = ana.plot_trendline(X, Y, compress=2)
    ana.remove_trendline(trendline=trendline)

    # coverage: figure passed
    fig, ax = misc.figure()
    trendline = ana.plot_trendline(X, Y, compress=2)
    ana.remove_trendline(fig=fig)
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_PLOT(mock_show):
    xdata = range(10)
    ydata = range(10)

    ana.PLOT(xdata, ydata)
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_plot_RMSD(mock_show):
    xdata = range(10)
    ydata = range(10)

    # coverage
    ana.plot_RMSD(RMSD_file=f"{pre}/RMSD.log", verbose=None)
    ana.plot_RMSD(RMSD_file=f"{pre}/RMSD.log", verbose=True, cut_min=0)
    ana.plot_RMSD(RMSD_file=f"{pre}/RMSD.log", verbose=False, title="title", save_as="./test.png")
    ana.plot_RMSD(RMSD_file=f"{pre}/RMSD.log", verbose=False, title="title", save_as="./test.png", filedir="./")
    misc.rm("./test.png")
    plt.close("all")
    return


def test_HELP_setup_bins():
    X = [0, 50, 100]

    bins, bins_step, cfg = ana._HELP_setup_bins(X, n_bins=2)
    assert np.all(bins == np.array([0., 50., 100.]))
    assert bins_step == 50.0
    assert isinstance(cfg, misc.CONFIG)

    bins, bins_step, cfg = ana._HELP_setup_bins(X, bins=[0, 50])
    assert np.all(bins == np.array([0, 50]))
    assert bins_step == 50
    assert isinstance(cfg, misc.CONFIG)
    return


@patch("matplotlib.pyplot.show")
def test_plot_hist(mock_show):
    ref = mda.Universe(pdb, tpr_resid_from_one=True)
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    FRAME, TIME, RMSD = ana.get_RMSD(mobile, ref, sel1="name CA", sel2="name CA")

    # coverage
    fig, ax = misc.figure()
    fig, ax, hist = ana.plot_hist([RMSD, RMSD, RMSD], ec=None, ax=ax, num=fig.number, fig=fig)   # multiple arrays, coverage of if cases
    fig, ax, hist = ana.plot_hist([RMSD, RMSD, RMSD], orientation="horizontal")
    fig, ax, hist = ana.plot_hist(RMSD, apply_cut_limits=True, orientation="horizontal")   # apply_cut_limits + orientation
    fig, ax, hist = ana.plot_hist(RMSD, apply_cut_limits=False, orientation="horizontal")  # apply_cut_limits + orientation
    fig, ax, hist = ana.plot_hist(RMSD, apply_cut_limits=True, orientation="vertical")     # apply_cut_limits + orientation
    fig, ax, hist = ana.plot_hist(RMSD, apply_cut_limits=False, orientation="vertical")    # apply_cut_limits + orientation
    fig, ax, hist = ana.plot_hist([RMSD, RMSD, RMSD], title="title", xlabel="xlabel", ylabel="ylabel", save_as="./temp.png")  # labels + savefig
    misc.rm("./temp.png")
    plt.close("all")
    return


@ patch("matplotlib.pyplot.show")
def test_plot_deltahist(mock_show):
    # coverage
    fig, ax = ana.plot_deltahist(RMSD_file=f"{pre}/RMSD2.log", RMSD_ref=f"{pre}/RMSD.log", logscale=True, orientation="horizontal", apply_cut_limits=True, show_all_hist=True)
    fig, ax = ana.plot_deltahist(RMSD_file=f"{pre}/RMSD2.log", RMSD_ref=f"{pre}/RMSD.log", logscale=False, orientation="horizontal", title="title", num=2)
    fig, ax = ana.plot_deltahist(RMSD_file=f"{pre}/RMSD2.log", RMSD_ref=f"{pre}/RMSD.log", logscale=True, orientation="vertical", save_as="./temp.png", apply_cut_limits=True)
    fig, ax = ana.plot_deltahist(RMSD_file=f"{pre}/RMSD2.log", RMSD_ref=f"{pre}/RMSD.log", logscale=False, orientation="vertical")
    misc.rm("./temp.png")
    plt.close("all")
    return


def test_HELP_convert_RMSD_nm2angstrom():
    RMSD_A = misc.read_file(f"{pre}/RMSD.log", usecols=1)
    RMSD_nm = RMSD_A/10

    val = ana._HELP_convert_RMSD_nm2angstrom(RMSD_nm)
    assert assert_allclose(val, RMSD_A) == None
    return


@ patch("matplotlib.pyplot.show")
def test_plot_HEATMAP(mock_show):
    data = np.random.randint(5, size=(5, 10))
    fig, ax = ana.plot_HEATMAP(data, title="title", xlabel="xlabel", ylabel="ylabel", cbar_label="cbar_label", save_as="./temp.pdf")
    misc.rm("./temp.pdf")
    plt.close("all")
    return


@ patch("matplotlib.pyplot.show")
def test_plot_HEATMAP_REX_RMSD(mock_show):
    # coverage
    fig, ax = ana.plot_HEATMAP_REX_RMSD(REX_RMSD_dir=f"{pre}/REX_RMSD_DIR", title="title", save_as="./temp.pdf")
    fig, ax = ana.plot_HEATMAP_REX_RMSD(REX_RMSD_dir=f"{pre}/REX_RMSD_DIR", auto_convert=False)
    misc.rm("./temp.pdf")
    plt.close("all")
    return
