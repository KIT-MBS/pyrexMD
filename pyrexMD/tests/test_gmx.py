# @Author: Arthur Voronin <arthur>
# @Date:   21.06.2021
# @Filename: test_gmx.py
# @Last modified by:   arthur
# @Last modified time: 23.06.2021

import pyrexMD.misc as misc
import pyrexMD.gmx as gmx
import pytest
import os
import shutil


cwd = misc.cwd(verbose=False)

# cwd is <...>/pyrexMD/tests/
if misc.realpath(f"{cwd}").split("/")[-1] == "tests":
    pre = "../examples/files/protein/"

# cwd is <...>/pyrexMD/tests/
elif misc.realpath(f"{cwd}").split("/")[-1] == "pyrexMD":
    pre = "./examples/files/protein/"

pdb = f"{pre}/1l2y.pdb"
tpr = f"{pre}/../traj/traj_protein.tpr"
xtc = f"{pre}/../traj/traj_protein.xtc"


def test_get_sel_code():
    assert gmx._get_sel_code("system") == "0"
    assert gmx._get_sel_code("protein") == "1"
    assert gmx._get_sel_code("ca") == "3"
    assert gmx._get_sel_code("bb") == "4"
    with pytest.raises(ValueError) as e_info:
        gmx._get_sel_code("unsupported_selection_string")
    return


def test_pdb2gmx():
    ref = gmx.get_ref_structure(pdb, ff='amber99sb-ildn', water='tip3p', ignh=True)
    o_file = gmx.pdb2gmx(f=ref, o="protein.gro", ff='amber99sb-ildn', water='tip3p', ignh=True)
    assert o_file == misc.realpath(f"{cwd}/protein.gro")
    return


def test_editconf():
    o_file = gmx.editconf(f="protein.gro", o="default", d=2.0, c=True, bt="cubic")
    assert o_file == misc.realpath(f"{cwd}/box.gro")

    o_file = gmx.editconf(f="protein.gro", o="box.gro", odir="./", d=2.0, c=True, bt="cubic")
    assert o_file == misc.realpath(f"{cwd}/box.gro")
    return


def test_convert_TPR2PDB():
    o_file = gmx.convert_TPR2PDB(tpr=tpr, o="default", d=2.0, c=True, bt="cubic")
    assert o_file == misc.realpath(f"{cwd}/traj_protein.pdb")

    o_file = gmx.convert_TPR2PDB(tpr="protein.gro", o="box.gro", odir="./", d=2.0, c=True, bt="cubic")
    assert o_file == misc.realpath(f"{cwd}/box.gro")
    return


def test_convert_TPR():
    o_file = gmx.convert_TPR(s=tpr, o="default")
    assert o_file == misc.realpath(f"{cwd}/traj_protein_protein.tpr")
    return


def test_solvate():
    o_file = gmx.solvate(cp="box.gro", o="solvent.gro")
    assert o_file == misc.realpath(f"{cwd}/solvent.gro")
    return


def test_grompp():
    o_file = gmx.grompp(f=f"{pre}/ions.mdp", o="ions.tpr", c="solvent.gro", maxwarn=2, cprint_color="red")
    o_file = gmx.grompp(f=f"{pre}/ions.mdp", o="ions.tpr", c="solvent.gro", maxwarn=2, cprint_color="red")  # coverage skip existing log
    assert o_file == misc.realpath(f"{cwd}/ions.tpr")
    return


def test_genion():
    misc.cp("topol.top", "topol_bak.top")  # make copy for later
    o_file = gmx.genion(s="ions.tpr", o="ions.gro", neutral=True, input="SOL")
    o_file = gmx.genion(s="ions.tpr", o="ions.gro", neutral=True, input="SOL")  # coverage skip existing log
    misc.rm("ions.gro")
    assert o_file == misc.realpath(f"{cwd}/ions.gro")
    return


def test_mdrun():
    misc.cp("topol_bak.top", "topol.top")
    gmx.genion(s="ions.tpr", o="ions.gro", neutral=True, input="SOL")
    gmx.grompp(f=f"{pre}/min.mdp", o="min.tpr", c="ions.gro")
    gmx.mdrun(deffnm="min", nsteps=1)
    return


def test_trjconv():
    o_file = gmx.trjconv(s=tpr, f=xtc, o="default", center=True)
    assert o_file == misc.realpath(f"{cwd}/traj_protein_protein.xtc")

    # coverage
    gmx.trjconv(s=tpr, f=xtc, o="traj_bb.xtc", sel="bb", center=True)
    with pytest.raises(ValueError) as e_info:
        gmx.trjconv(s=pdb, f=xtc, o="traj_bb.xtc", sel="bb", center=True)
    return


def test_fix_TRAJ():
    tpr_file, xtc_file = gmx.fix_TRAJ(tpr=tpr, xtc=xtc)
    assert tpr_file == misc.realpath("traj_protein_protein.tpr")
    assert xtc_file == misc.realpath("traj_protein_protein.xtc")

    tpr_file, xtc_file = gmx.fix_TRAJ(tpr=tpr, xtc=xtc, o=["fix_TRAJ.tpr", "fix_TRAJ.xtc"])
    assert tpr_file == misc.realpath("fix_TRAJ.tpr")
    assert xtc_file == misc.realpath("fix_TRAJ.xtc")

    # coverage
    with pytest.raises(ValueError) as e_info:
        gmx.fix_TRAJ(tpr=tpr, xtc=xtc, o="NOT_A_LIST")
    return


def test_get_RMSD():
    ref = gmx.get_ref_structure(tpr, ff='amber99sb-ildn', water='tip3p', ignh=True)
    o_file = gmx.get_RMSD(ref, xtc)
    assert o_file == misc.realpath("rmsd.xvg")
    # coverage
    with pytest.raises(ValueError) as e_info:
        o_file = gmx.get_RMSD(pdb, xtc)
    return


def test_create_complex():
    gro = "min.gro"
    o_file = gmx.create_complex([gro, gro], o="complex.gro")
    assert o_file == misc.realpath("complex.gro")

    # coverage
    with pytest.raises(misc.Error) as e_info:
        gmx.create_complex(pdb, o="complex.gro")
    with pytest.raises(misc.Error) as e_info:
        gmx.create_complex([pdb], o="complex.gro")
    with pytest.raises(misc.Error) as e_info:
        gmx.create_complex([pdb, pdb], o="complex.gro")

    return


def test_extend_complex_topology():
    ligand_name = "random_name1"
    ligand_itp = f"{pre}/../atp/atp.itp"
    ligand_prm = f"{pre}/../atp/atp.prm"
    ligand_nmol = 2
    o_file = gmx.extend_complex_topology(ligand_name=ligand_name,
                                         ligand_itp=ligand_itp,
                                         ligand_prm=ligand_prm,
                                         ligand_nmol=ligand_nmol,
                                         top="topol.top",
                                         top_out="topol_complex.top")
    assert o_file == misc.realpath("topol_complex.top")

    # coverage: add 2nd ligand
    ligand_name = "random_name2"
    o_file = gmx.extend_complex_topology(ligand_name=ligand_name,
                                         ligand_itp=ligand_itp,
                                         ligand_prm=ligand_prm,
                                         ligand_nmol=ligand_nmol,
                                         top="topol_complex.top",
                                         top_out="topol_complex.top")
    assert o_file == misc.realpath("topol_complex.top")
    return


def test_create_positions_dat():
    # coverage
    file_dir = gmx.create_positions_dat()
    file_dir = gmx.create_positions_dat(nmol=100)
    with pytest.raises(TypeError) as e_info:
        file_dir = gmx.create_positions_dat(nmol="100")
    return


def test_get_templates():
    # coverage
    gmx.get_template_1_EM(gmx_mpi=True)
    gmx.get_template_1_EM(gmx_mpi=False)

    gmx.get_template_2_NVT(gmx_mpi=True)
    gmx.get_template_2_NVT(gmx_mpi=False)

    gmx.get_template_3_NPT(gmx_mpi=True)
    gmx.get_template_3_NPT(gmx_mpi=False)

    gmx.get_template_4_MD(gmx_mpi=True)
    gmx.get_template_4_MD(gmx_mpi=False)
    return


def test_clean_up():
    gmx.clean_up()
    gmx.clean_up(path="*.pdb", pattern=None, ignore=".pdb")            # coverage
    gmx.clean_up(path="./", pattern="*.pdb", ignore=[".pdb", ".gro"])  # coverage
    gmx.clean_up(path="./", pattern="*.pdb")
    gmx.clean_up(path="./", pattern="*.gro")
    gmx.clean_up(path="./", pattern="*.top")
    gmx.clean_up(path="./", pattern="*.tpr")
    gmx.clean_up(path="./", pattern="*.xtc")
    gmx.clean_up(path="./", pattern="*.xvg")
    gmx.clean_up(path="./", pattern="*.edr")
    gmx.clean_up(path="./", pattern="*.itp")
    gmx.clean_up(path="./", pattern="*.log")
    gmx.clean_up(path="./", pattern="*.trr")
    gmx.clean_up(path="./", pattern="*.mdp")
    return


# clean up at after tests
def test_clean_up_after_tests():
    if os.path.exists('./logs'):
        shutil.rmtree('./logs')
    if os.path.exists('./positions_dat'):
        shutil.rmtree('./positions_dat')
    return
