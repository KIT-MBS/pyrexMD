# @Author: Arthur Voronin <arthur>
# @Date:   22.06.2021
# @Filename: test_rex.py
# @Last modified by:   arthur
# @Last modified time: 01.07.2021


import pyrexMD.misc as misc
import pyrexMD.rex as rex
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
pre = f"{main_dir}/examples/files/rex/"
pre2 = f"{main_dir}/tests/files/"
score_fin = f"{pre}/1LMB.rr"
pdb = f"{pre}/1lmb_Chain4.pdb"
pdb2 = f"{main_dir}/examples/files/traj/2hba_ChainB_ref.pdb"


def test_apply_ff_best_decoys():
    decoy_dir = f"{pre}/decoys/"
    n_decoys = 2    # use only 2 for test
    odir = rex.apply_ff_best_decoys(decoy_dir, n_decoys=n_decoys, verbose=True)
    assert odir == misc.realpath("./1LMB_best_decoys_ref")

    # coverage
    rex.apply_ff_best_decoys(decoy_dir, n_decoys=1, verbose=True, pdbid="1LMB")
    return


def test_assign_best_decoys():
    best_decoys_dir = f"{pre}/1LMB_best_decoys_ref"
    rex.assign_best_decoys(best_decoys_dir)
    return


def test_get_REX_DIRS():
    REX_DIRS = rex.get_REX_DIRS(realpath=False)
    assert len(REX_DIRS) != 0
    return


def test_get_REX_PDBS():
    REX_PDBS = rex.get_REX_PDBS(realpath=False)
    assert len(REX_PDBS) != 0
    return


def test_check_REX_PDBS():
    REX_PDBS = rex.get_REX_PDBS(realpath=False)

    # check pass
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=pdb, verbose=True)
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=None, verbose=True)

    # check fail
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=pdb2, verbose=True)
    return


def test_check2_REX_PDBS():
    REX_PDBS = rex.get_REX_PDBS(realpath=False)

    # check pass
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=pdb, verbose=True)
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=None, verbose=True)

    # check fail
    rex.check_REX_PDBS(REX_PDBS=REX_PDBS, ref_pdb=pdb2, verbose=True)
    return


def test_WF_getParameter_boxsize():
    logfile = f"{pre}/important_files/editconf_1.log"
    boxsize = rex.WF_getParameter_boxsize(logfile=logfile)
    assert boxsize == 9.4

    # coverage
    with pytest.raises(misc.Error):
        logfile = f"{pre}/important_files/solvate_1.log"
        boxsize = rex.WF_getParameter_boxsize(logfile=logfile)
    return


def test_WF_getParameter_maxsol():
    logfile = f"{pre}/important_files/solvate_1.log"
    maxsol = rex.WF_getParameter_maxsol(logfile=logfile)
    assert maxsol == 26042

    # coverage
    with pytest.raises(misc.Error):
        logfile = f"{pre}/important_files/editconf_1.log"
        maxsol = rex.WF_getParameter_maxsol(logfile=logfile)
    return


def test_WF_get_system_parameters():
    boxsize, maxsol = rex.WF_get_system_parameters(wdir="./rex_0_get_system_parameters")
    assert boxsize == 9.4
    assert 26200 > maxsol > 26000
    return


def test_WF_REX_setup():
    # copy mdp files temporarily
    misc.cp(f"{pre}/*mdp", ".")

    rex_dirs = rex.get_REX_DIRS()[:2]    # use only 2 rex dirs for test
    log1 = f"{pre}/important_files/editconf_1.log"
    boxsize = rex.WF_getParameter_boxsize(logfile=log1)
    log2 = f"{pre}/important_files/solvate_1.log"
    maxsol = rex.WF_getParameter_maxsol(logfile=log2)

    rex.WF_REX_setup(rex_dirs=rex_dirs, boxsize=boxsize, maxsol=maxsol)
    return


def test_WF_REX_setup_energy_minimization():
    rex_dirs = rex.get_REX_DIRS()[:2]    # use only 2 rex dirs for test
    rex.WF_REX_setup_energy_minimization(rex_dirs=rex_dirs, nsteps=5, verbose=True)
    return


def test_create_special_group_ndx():
    rex.create_special_group_ndx(ref_pdb=pdb, sel="name CA", save_as="./special_group.ndx")
    with open(f"{pre2}/special_group.ndx", "r") as expected, open("./special_group.ndx", "r") as val:
        lines1 = expected.readlines()
        lines2 = val.readlines()
        assert lines1 == lines2
    misc.rm("./special_group.ndx")
    return


def test_create_pull_group_ndx():
    rex.create_pull_group_ndx(ref_pdb=pdb, sel="name CA", save_as="./pull_group.ndx")
    with open(f"{pre2}/pull_group.ndx", "r") as expected, open("./pull_group.ndx", "r") as val:
        lines1 = expected.readlines()
        lines2 = val.readlines()
        assert lines1 == lines2
    misc.rm("./pull_group.ndx")
    return


def test_prep_REX_temps():
    rex.prep_REX_temps(T_0=280, n_REX=50, k=0.005)
    with open(f"{pre2}/rex_temps.log", "r") as expected, open("./rex_temps.log", "r") as val:
        lines1 = expected.readlines()
        lines2 = val.readlines()
        assert lines1 == lines2

    # coverage
    rex.prep_REX_temps(T_0=280, n_REX=12, k=0.005)
    # do not remove rex_temps.log -> required later
    return


def test_prep_REX_mdp():
    rex.prep_REX_mdp(n_REX=2)   # use only 2 replicas for test
    return


def test_prep_REX_tpr():
    # use only 2 replicas for test and topol.top instead of topol_mod.top
    rex.prep_REX_tpr(n_REX=2, f="rex.mdp", o="rex.tpr", c="em.gro", p="topol.top")
    return


# clean up at after tests
def test_clean_up_after_tests():
    misc.rm("*.mdp")
    misc.rm("topol.top")
    misc.rm("posre.itp")
    misc.rm("rex_temps.log")

    shutil.rmtree('./1LMB_best_decoys_ref')
    shutil.rmtree("./rex_0_get_system_parameters")
    for dir in rex.get_REX_DIRS():
        shutil.rmtree(dir)
    return
