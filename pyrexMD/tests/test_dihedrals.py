# @Author: Arthur Voronin <arthur>
# @Date:   22.06.2021
# @Filename: test_dihedrals.py
# @Last modified by:   arthur
# @Last modified time: 01.07.2021


import pyrexMD.misc as misc
import pyrexMD.analysis.dihedrals as dih
import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import patch
import pathlib
import pytest
import os


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


@patch("matplotlib.pyplot.show")
def test_get_ramachandran(mock_show):
    u = mda.Universe(tpr, xtc)
    expected_angles = np.load(f"{pre}/rama/rama_angles.npy")
    expected_frames = np.load(f"{pre}/rama/rama_frames.npy")
    expected_times = np.load(f"{pre}/rama/rama_times.npy")

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
    expected_angles = np.load(f"{pre}/janin/janin_angles.npy")
    expected_frames = np.load(f"{pre}/janin/janin_frames.npy")
    expected_times = np.load(f"{pre}/janin/janin_times.npy")

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
    expected = np.load(f"{pre}/dihedrals/phi.npy")
    val = dih.get_phi_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_psi_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(f"{pre}/dihedrals/psi.npy")
    val = dih.get_psi_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_omega_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(f"{pre}/dihedrals/omega.npy")
    val = dih.get_omega_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_chi1_values():
    mobile = mda.Universe(tpr, xtc)
    expected = np.load(f"{pre}/dihedrals/chi1.npy")
    val = dih.get_chi1_values(mobile, sel="protein")
    assert assert_allclose(val, expected) == None
    return


def test_get_chi2_values():
    # coverage: 1l2y has no chi2
    mobile = mda.Universe(tpr, xtc)
    with pytest.raises(AttributeError):
        val = dih.get_chi2_values(mobile, sel="protein")
    return
