# @Author: Arthur Voronin <arthur>
# @Date:   21.05.2021
# @Filename: test_topology.py
# @Last modified by:   arthur
# @Last modified time: 23.06.2021


import pyrexMD.misc as misc
import pyrexMD.topology as top
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose
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


def test_get_resids_shift():
    ref = mda.Universe(pdb)
    ref_shifted = mda.Universe(pre + "1l2y_ref_shifted.pdb", tpr_resid_from_one=False)
    assert top.get_resids_shift(ref_shifted, ref) == 4

    # coverage
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=False)
    assert top.get_resids_shift(mobile, mobile) == 0
    return


def test_shift_resids():
    ref = mda.Universe(pdb)
    expected = ref.residues.resids + 2

    top.shift_resids(ref, 2)
    val = ref.residues.resids
    assert assert_allclose(val, expected) == None

    # coverage
    top.shift_resids(ref, None)
    return


def test_align_resids():
    ref = mda.Universe(pdb)
    mobile = mda.Universe(tpr, xtc)

    if np.all(ref.residues.resids != mobile.residues.resids):
        expected = np.array(ref.residues.resids)
    top.align_resids(mobile, ref)
    assert assert_allclose(expected, ref.residues.resids) == None
    assert assert_allclose(expected, mobile.residues.resids) == None
    return


def test_HELP_print():
    # coverage
    top.__HELP_print(info='', verbose=True)
    top.__HELP_print(info='', verbose=False)
    top.__HELP_print(info='blubb', verbose=True)
    return


def test_norm_ids():
    ref = mda.Universe(pdb)
    expected = ref.atoms.ids
    ref.atoms.ids += 5

    if np.all(ref.atoms.ids != expected):
        top.norm_ids(ref)
    assert assert_allclose(ref.atoms.ids, expected) == None

    # coverage
    with pytest.raises(TypeError) as e_info:
        top.norm_ids("wrong_type")
    return


def test_norm_resids():
    ref = mda.Universe(pdb)
    expected = ref.residues.resids
    ref.residues.resids += 5

    if np.all(ref.residues.resids != expected):
        top.norm_resids(ref)
    assert assert_allclose(ref.residues.resids, expected) == None

    # coverage
    with pytest.raises(TypeError) as e_info:
        top.norm_resids("wrong_type")
    return


def test_norm_and_align_universe():
    # case1: mobile not normed
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    expected_resids = ref.residues.resids
    expected_ids = ref.atoms.ids

    mobile.residues.resids -= 10
    mobile.atoms.ids -= 50

    if np.all(mobile.residues.resids != expected_resids) or np.all(mobile.atoms.ids != expected_ids):
        top.norm_and_align_universe(mobile, ref)

    assert assert_allclose(mobile.residues.resids, expected_resids) == None
    assert assert_allclose(mobile.atoms.ids, expected_ids) == None
    assert assert_allclose(ref.residues.resids, expected_resids) == None
    assert assert_allclose(ref.atoms.ids, expected_ids) == None

    # case 2: ref not normed
    mobile = mda.Universe(tpr, xtc, tpr_resid_from_one=True)
    ref = mda.Universe(pdb)
    expected_resids = ref.residues.resids
    expected_ids = ref.atoms.ids

    ref.residues.resids -= 10
    ref.atoms.ids -= 50

    if np.all(mobile.residues.resids != expected_resids) or np.all(mobile.atoms.ids != expected_ids):
        top.norm_and_align_universe(mobile, ref)

    assert assert_allclose(mobile.residues.resids, expected_resids) == None
    assert assert_allclose(mobile.atoms.ids, expected_ids) == None
    assert assert_allclose(ref.residues.resids, expected_resids) == None
    assert assert_allclose(ref.atoms.ids, expected_ids) == None
    ############################################################################
    ### coverage
    ref = mda.Universe(pdb)
    ref_shifted = mda.Universe(pdb)
    # len of ids do not match
    with pytest.raises(ValueError) as e_info:
        top.norm_and_align_universe(ref, ref_shifted.select_atoms("resid 1-10"))

    # len of resids do not match
    with pytest.raises(ValueError) as e_info:
        top.norm_and_align_universe(ref.select_atoms("name CA and resid 1-3"), ref_shifted.select_atoms("index 1 3 25"))

    # ref and mobile resids do not match
    ref_shifted.residues.resids += 10
    top.norm_and_align_universe(ref, ref_shifted)

    # ref and mobile ids do not match
    ref_shifted.atoms.ids += 50
    top.norm_and_align_universe(ref, ref_shifted)
    return


def test_true_select_atoms():
    ref = mda.Universe(pdb)

    a0 = ref.select_atoms("protein and name CA")
    a1 = top.true_select_atoms(pdb, sel="protein and name CA")
    a2 = top.true_select_atoms(ref, sel="protein and name CA")

    assert assert_allclose(a0.atoms.ids, a1.atoms.ids) == None
    assert assert_allclose(a0.residues.resids, a1.residues.resids) == None
    assert assert_allclose(a1.atoms.ids, a2.atoms.ids) == None
    assert assert_allclose(a1.residues.resids, a2.residues.resids) == None

    # coverage: fetch online
    top.true_select_atoms("1l2y", sel="protein")
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


def test_sel2selid():
    ref = mda.Universe(pdb)
    sel = "protein and name CA"
    selid = top.sel2selid(ref, sel)
    assert (ref.select_atoms(sel) == ref.select_atoms(selid))

    ref = mda.Universe(pdb)
    sel = "type O"
    selid = top.sel2selid(ref, sel)
    assert (ref.select_atoms(sel) == ref.select_atoms(selid))

    # coverage
    selid = top.sel2selid(pdb, sel="nucleic")
    rna = top.true_select_atoms("4enc", sel="nucleic")
    selid = top.sel2selid(rna, sel="nucleic")
    return


def test_dump_structure():
    ref = mda.Universe(pdb)
    dirpath = top.dump_structure(ref, frames=0, save_as="./temp.pdb")
    assert dirpath == misc.realpath(".")

    # coverage
    with pytest.raises(misc.Error) as e_info:
        top.dump_structure(ref, frames=0, save_as="temp")
    misc.rm("./temp_0.pdb")
    return


def test_parse_PDB():
    ref = mda.Universe(pdb)
    RESID, RESNAME, ID, NAME = top.parsePDB(pdb, sel="protein", norm=True)

    assert assert_allclose(RESID, ref.atoms.resids) == None
    assert assert_allclose(ID, ref.atoms.ids) == None
    assert np.all(RESNAME == ref.atoms.resnames)  # strings have no tolerance
    assert np.all(NAME == ref.atoms.names)        # strings have no tolerance

    # coverage
    top.parsePDB(pre + "../4enc_cluster_01.pdb", sel="nucleic")
    return


def test_DCA_rex2atom_mapping():
    ref_pdb = pre + "../2hba/2hba_ChainB_ref.pdb"
    DCA_fin = pre + "../2hba/2hba.score"
    n_DCA = 40
    usecols = (0, 1)

    # TEST: filter DCA is False
    read = misc.read_file(DCA_fin, usecols=usecols, n_rows=None)
    read_zip = zip(read[0], read[1])
    expected_RES_PAIR = [(int(item[0]), int(item[1])) for item in read_zip]

    RES_PAIR, ATOM_PAIR = top.DCA_res2atom_mapping(ref_pdb=ref_pdb, DCA_fin=DCA_fin, n_DCA=n_DCA, usecols=usecols, pdbid="2hba", save_as="./2HBA_DCA_used.log", filter_DCA=False)
    # do not remove log -> will be required in test_DCA_modify_topology()
    assert assert_allclose(RES_PAIR, expected_RES_PAIR[:n_DCA]) == None

    # TEST: filter DCA is True
    read = misc.read_file(DCA_fin, usecols=usecols, n_rows=None)
    read_zip = zip(read[0], read[1])
    expected_RES_PAIR = [(int(item[0]), int(item[1])) for item in read_zip if abs(item[0]-item[1]) >= 4]

    RES_PAIR, ATOM_PAIR = top.DCA_res2atom_mapping(ref_pdb=ref_pdb, DCA_fin=DCA_fin, n_DCA=n_DCA, usecols=usecols, pdbid="2hba", save_as="./2HBA_DCA_used.log", filter_DCA=True)
    # do not remove log -> will be required in test_DCA_modify_topology()
    assert assert_allclose(RES_PAIR, expected_RES_PAIR[:n_DCA]) == None

    return


def test_DCA_modify_topology():
    top_fin = pre + "../2hba/2hba_topol.top"
    DCA_used_fin = "./2HBA_DCA_used.log"
    n_DCA = 40

    top.DCA_modify_topology(top_fin=top_fin, DCA_used_fin=DCA_used_fin, n_DCA=None, pdbid="2hba", save_as="./2hba_topol_mod.top")
    with open(pre + "../2hba/2hba_topol_mod.top", "r") as expected, open("./2hba_topol_mod.top", "r") as val:
        lines1 = expected.readlines()
        lines2 = val.readlines()
        assert lines1 == lines2
    misc.rm("./2HBA_DCA_used.log")
    misc.rm("./2hba_topol_mod.top")
    return


def test_modify_scorefile():
    DCA_fin = pre + "../2hba/2hba.score"
    n_DCA = 40
    usecols = (0, 1)

    I, J = misc.read_file(DCA_fin, usecols=usecols, n_rows=None)
    expected_I = I+10
    expected_J = J+10

    new_file = top.DCA_modify_scoreFile(score_fin=DCA_fin, shift_res=10, res_cols=usecols, score_col=5)
    I, J = misc.read_file(new_file, usecols=usecols, n_rows=None)

    assert assert_allclose(I, expected_I) == None
    assert assert_allclose(J, expected_J) == None
    return

#
# def test_clean_up_files():
#     rm("./2HBA_DCA_used.txt")
#     return
