# @Author: Arthur Voronin <arthur>
# @Date:   09.05.2021
# @Filename: test_contacts.py
# @Last modified by:   arthur
# @Last modified time: 01.07.2021

import pyrexMD.misc as misc
import pyrexMD.analysis.contacts as con
import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch
import pathlib
import pytest
import os
import shutil


# find main directory of pyrexMD
posixpath = pathlib.Path(".").rglob("*core.py")   # generator for matching paths
pathname = posixpath.send(None).as_posix()        # get first path name
main_dir = misc.relpath(misc.realpath(pathname).rstrip("core.py"))  # main directory of pyrexMD

# set up test paths
cwd = misc.cwd(verbose=False)
pre = f"{main_dir}/tests/files/1l2y"
pdb = f"{pre}/1l2y_ref.pdb"
tpr = f"{pre}/traj.tpr"
xtc = f"{pre}/traj.xtc"


def test_get_Native_Contacts():
    ref = mda.Universe(pdb)
    NC, NC_d = con.get_Native_Contacts(ref, sel="protein", d_cutoff=6.0, ignh=True)

    assert (np.array(NC) == np.load(f"{pre}/NC.npy")).all()
    assert (np.array(NC_d) == np.load(f"{pre}/NC_d.npy")).all()

    # coverage
    con.get_Native_Contacts(pdb, sel="protein", ignh=False, method=1, save_as="./temp.txt")
    misc.rm("./temp.txt")
    return


@patch("matplotlib.pyplot.show")
def test_get_NC_distances(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    NC_expected = np.load(f"{pre}/NC.npy")
    NC_dist_expected = np.load(f"{pre}/NC_dist.npy")
    DM_expected = np.load(f"{pre}/DM.npy")

    NC, NC_dist, DM = con.get_NC_distances(mobile, ref)
    assert np.all(NC == NC_expected)
    assert assert_allclose(NC_dist, NC_dist_expected) == None
    assert assert_allclose(DM, DM_expected) == None

    # coverage
    con.get_NC_distances(pdb, pdb, plot=True, save_as="./temp.png")
    misc.rm("./temp.png")
    con.get_NC_distances(pdb, pdb, method="contact_distance")
    with pytest.raises(ValueError):
        con.get_NC_distances(mobile, ref, method="value_raising_error")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_BC_distances(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)

    # use native contacts as bias contacts -> produces same values
    BC_expected = np.load(f"{pre}/NC.npy")
    BC_dist_expected = np.load(f"{pre}/NC_dist.npy")
    DM_expected = np.load(f"{pre}/DM.npy")

    BC, BC_dist, DM = con.get_BC_distances(mobile, bc=BC_expected)
    assert np.all(BC == BC_expected)
    assert assert_allclose(BC_dist, BC_dist_expected) == None
    assert assert_allclose(DM, DM_expected) == None

    # coverage
    con.get_BC_distances(mobile, bc=BC_expected, method="contact_distance", save_as="./temp.txt", plot=True)
    misc.rm("./temp.txt")
    with pytest.raises(ValueError):
        con.get_BC_distances(mobile.filename, bc=BC_expected, method="value_raising_error")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_plot_Contact_Map(mock_show):
    ref = mda.Universe(pdb)

    # coverage
    fig, ax = con.plot_Contact_Map(ref, DCA_fin=f"{pre}/1l2y_mixed_contacts.txt", ignh=True)
    fig, ax = con.plot_Contact_Map(pdb, ignh=False, save_plot="./temp.png")
    misc.rm("./temp.png")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_plot_Contact_Map_Distances(mock_show):
    ref = mda.Universe(pdb)
    NC = np.load(f"{pre}/NC.npy")
    NC_dist = np.load(f"{pre}/NC_dist.npy")

    # coverage
    fig, ax = con.plot_Contact_Map_Distances(ref, NC, NC_dist, save_plot="./temp.png")
    misc.rm("./temp.png")
    fig, ax = con.plot_Contact_Map_Distances(pdb, NC, NC_dist, title="title")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_plot_DCA_TPR(mock_show):
    ref = mda.Universe(pdb)
    DCA_fin = f"{pre}/1l2y_mixed_contacts.txt"

    fig, ax = con.plot_DCA_TPR(ref, DCA_fin=DCA_fin, n_DCA=20, save_plot="./temp.png", save_log="./temp.log")
    misc.rm("./temp.png")
    misc.rm("./temp.log")

    # coverage
    fig, ax = con.plot_DCA_TPR(pdb, DCA_fin=DCA_fin, n_DCA=20, ignh=False, TPR_layer="bg")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_QNative(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    FRAMES, QNATIVE = con.get_QNative(mobile, ref, sel="protein and name CA", d_cutoff=6.0, plot=True)

    assert (FRAMES == np.load(f"{pre}/FRAMES.npy")).all()
    assert assert_allclose(QNATIVE, np.load(f"{pre}/QNATIVE.npy")) == None

    # coverage
    con.get_QNative(mobile, ref, sel="protein and name CA", d_cutoff=6.0, plot=True, save_plot=True, save_as="./temp.png")
    misc.rm("./temp.png")
    with pytest.raises(misc.Error):
        con.get_QNative(mobile, ref, sel1="backbone")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_QBias(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    bc = np.load(f"{pre}/NC.npy")
    FRAMES, QBIAS, CM = con.get_QBias(mobile, bc, d_cutoff=6.0, plot=False, softcut=False)

    assert (FRAMES == np.load(f"{pre}/FRAMES.npy")).all()
    assert assert_allclose(QBIAS, np.load(f"{pre}/QBIAS.npy"), atol=0.001) == None
    assert (CM == np.load(f"{pre}/CM.npy")).all()

    # coverage
    con.get_QBias(mobile, bc, d_cutoff=6.0, plot=True, softcut=True, include_selfcontacts=True, save_plot=True, save_as="./temp.png")
    misc.rm("./temp.png")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_QBias_TPFP(mock_show):
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    NC = np.load(f"{pre}/NC.npy")
    NC = [tuple(item) for item in NC]
    BC = list(NC)
    BC.append((12, 18))  # get_QBias_TPFP requires atleast 1 non-native contact

    FRAMES, TP, FP, CM = con.get_QBias_TPFP(mobile, BC=BC, NC=NC, plot=True, include_selfcontacts=False)

    # coverage
    con.get_QBias_TPFP(mobile, BC=BC, NC=NC, include_selfcontacts=True, prec=3)
    # with pytest.raises(misc.Error):
    #     con.get_QBias_TPFP(mobile, BC=np.array(BC), NC=NC)
    #     con.get_QBias_TPFP(mobile, BC=BC, NC=np.array(NC))
    return


def test_get_formed_contactpairs():
    u = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    cm = np.load(f"{pre}/CM.npy")[-1]   # was tested with last frame
    CP = con.get_formed_contactpairs(u, cm)

    assert (CP == np.load(f"{pre}/CP.npy")).all()

    # coverage
    con.get_formed_contactpairs(u, cm, include_selfcontacts=True)
    with pytest.raises(misc.Error):
        con.get_formed_contactpairs(u, cm.reshape((20, 4, 5)))
    return


# clean up at after tests
def test_clean_up_after_tests():
    if os.path.exists('./plots'):
        shutil.rmtree('./plots')
    if os.path.exists('./logs'):
        shutil.rmtree('./logs')
    return
