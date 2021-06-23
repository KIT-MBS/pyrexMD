# @Author: Arthur Voronin <arthur>
# @Date:   22.06.2021
# @Filename: test_dihedrals.py
# @Last modified by:   arthur
# @Last modified time: 23.06.2021


import pyrexMD.misc as misc
import pyrexMD.analysis.dihedrals as dih
import MDAnalysis as mda
import matplotlib.pyplot as plt
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
elif misc.realpath(f"{cwd}").split("/")[-1] == "pyrexMD":
    pre = "./tests/files/1l2y/"

pdb = pre + "1l2y_ref.pdb"
tpr = pre + "traj.tpr"
xtc = pre + "traj.xtc"


@patch("matplotlib.pyplot.show")
def test_get_ramachandran(mock_show):
    u = mda.Universe(tpr, xtc)
    expected_angles = np.load(pre + "rama/rama_angles.npy")
    expected_frames = np.load(pre + "rama/rama_frames.npy")
    expected_times = np.load(pre + "rama/rama_times.npy")

    val = dih.get_ramachandran(u, sel="protein", plot=True)
    assert assert_allclose(val.angles, expected_angles) == None
    assert assert_allclose(val.frames, expected_frames) == None
    assert assert_allclose(val.times, expected_times) == None

    # coverage
    dih.get_ramachandran(u, sel="protein", plot=True, ref=True, num=1)
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_get_janin(mock_show):
    u = mda.Universe(tpr, xtc)
    expected_angles = np.load(pre + "janin/janin_angles.npy")
    expected_frames = np.load(pre + "janin/janin_frames.npy")
    expected_times = np.load(pre + "janin/janin_times.npy")

    val = dih.get_janin(u, sel="protein", plot=True)
    assert assert_allclose(val.angles, expected_angles) == None
    assert assert_allclose(val.frames, expected_frames) == None
    assert assert_allclose(val.times, expected_times) == None

    # coverage
    dih.get_janin(u, sel="protein", plot=True, ref=True, num=1)
    plt.close("all")
    return


def test_get_phi_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(pre + "dihedrals/phi.npy")
    val = dih.get_phi_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_psi_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(pre + "dihedrals/psi.npy")
    val = dih.get_psi_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_omega_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(pre + "dihedrals/omega.npy")
    val = dih.get_omega_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_chi1_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(pre + "dihedrals/chi1.npy")
    val = dih.get_chi1_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_chi2_values():
    # coverage: 1l2y has no chi2
    mobile = mda.Universe(tpr, xtc)
    with pytest.raises(AttributeError) as e_info:
        val = dih.get_chi2_values(mobile, sel="protein")
    return
