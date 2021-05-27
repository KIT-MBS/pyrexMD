# @Author: Arthur Voronin <arthur>
# @Date:   09.05.2021
# @Filename: test_contacts.py
# @Last modified by:   arthur
# @Last modified time: 27.05.2021

import pyrexMD.analysis.contacts as con
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose

pre = "./files/1l2y/"
pdb = pre + "1l2y_ref.pdb"
tpr = pre + "traj.tpr"
xtc = pre + "traj.xtc"


def test_get_Native_Contacts():
    ref = mda.Universe(pdb)
    NC, NC_d = con.get_Native_Contacts(ref, sel="protein", d_cutoff=6.0, ignh=True)

    assert (np.array(NC) == np.load(pre + "NC.npy")).all()
    assert (np.array(NC_d) == np.load(pre + "NC_d.npy")).all()
    return


def test_get_QNative():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    FRAMES, QNATIVE = con.get_QNative(mobile, ref, sel="protein and name CA", d_cutoff=6.0, plot=False)

    assert (FRAMES == np.load(pre + "FRAMES.npy")).all()
    assert assert_allclose(QNATIVE, np.load(pre + "QNATIVE.npy")) == None
    return


def test_get_QBias():
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    bc = np.load(pre + "NC.npy")
    FRAMES, QBIAS, CM = con.get_QBias(mobile, bc, d_cutoff=6.0, plot=False)

    assert (FRAMES == np.load(pre + "FRAMES.npy")).all()
    assert assert_allclose(QBIAS, np.load(pre + "QBIAS.npy")) == None
    assert (CM == np.load(pre + "CM.npy")).all()
    return


def test_get_formed_contactpairs():
    u = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    cm = np.load(pre + "CM.npy")[-1]   # was tested with last frame
    CP = con.get_formed_contactpairs(u, cm)

    assert (CP == np.load(pre + "CP.npy")).all()
    return
