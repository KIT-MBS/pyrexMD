# @Author: Arthur Voronin <arthur>
# @Date:   10.05.2021
# @Filename: test_analysis.py
# @Last modified by:   arthur
# @Last modified time: 21.05.2021


import pyrexMD.analysis.analysis as ana
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import pytest

pre = "./files/1l2y/"
pdb = pre + "1l2y_ref.pdb"
tpr = pre + "traj.tpr"
xtc = pre + "traj.xtc"


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


def test_get_RMSD():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    val = ana.get_RMSD(mobile, ref, sel1='backbone', sel2='backbone')
    expected = np.load(pre + "get_RMSD.npy")
    assert assert_allclose(val[0], expected[0]) == None
    assert assert_allclose(val[1], expected[1]) == None
    assert assert_allclose(val[2], expected[2]) == None
    return


def test_get_RMSF():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    val = ana.get_RMSF(mobile, "backbone")
    expected = np.load(pre + "get_RMSF.npy")
    assert assert_allclose(val, expected) == None
    return


def test_get_Distance_Matrices():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    val = ana.get_Distance_Matrices(mobile)
    expected = np.load(pre + "get_Distance_Matrices.npy")
    assert assert_allclose(val, expected) == None
    return


def text_shortest_RES_distances():
    ref = mda.Universe(pdb)
    val = ana.shortest_RES_distances(ref)
    expected = np.load(pre + "shorest_RES_distances.npy")
    assert assert_allclose(val[0], expected[0]) == None
    assert (val[1] == expected[1]).all()  # this could cause problems as mixed types
    return
