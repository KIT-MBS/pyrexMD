# @Author: Arthur Voronin <arthur>
# @Date:   21.05.2021
# @Filename: test_topology.py
# @Last modified by:   arthur
# @Last modified time: 21.05.2021


import pyrexMD.topology as top
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose
import pytest

pre = "./files/1l2y/"
pdb = pre + "1l2y_ref.pdb"
tpr = pre + "traj.tpr"
xtc = pre + "traj.xtc"


def test_get_resids_shift():
    ref = mda.Universe(pdb)
    ref_shifted = mda.Universe(pre + "1l2y_ref_shifted.pdb", tpr_resid_from_one=False)
    assert top.get_resids_shift(ref_shifted, ref) == 4
    return


def test_get_matching_selection():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=False)
    ref = mda.Universe(pdb)
    val = top.get_matching_selection(mobile, ref, "backbone", norm=False)
    expected = ('backbone and resid 0-19', 'backbone')
    assert val == expected

    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    val = top.get_matching_selection(mobile, ref, "backbone", norm=True)
    expected = ('backbone and resid 1-20', 'backbone')
    assert val == expected
    return
