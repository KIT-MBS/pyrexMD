import myPKG.misc as misc
import gromacs as gmx

### set which gmx command should be used (gmx or gmx_mpi)
# gmx_cmd = "gmx"
gmx_cmd = "gmx_mpi"


def convert_PDB(tpr, save_as="default", verbose=True):
    """
    Args:
        tpr (str): <rex_protein.tpr>
        save_as (str):
            "default": rex.tpr -> rex.pdb
        verbose (bool)

    Returns:
        pdb_file (str): realpath of converted pdb file
    Example:
        gmx code:
            gmx editconf -f rex_protein.tpr -o rex_protein.pdb
        python code:
            convert_PDB(tpr="rex_protein.tpr", save_as="rex_protein.pdb")

    """
    if save_as == "default":
        pre = misc.get_base(tpr)
        save_as = f"{pre}.pdb"
    misc.bash_cmd(f"{gmx_cmd} editconf -f {tpr} -o {save_as}")

    pdb_file = misc.realpath(save_as)
    if verbose:
        print("Saved converted pdb as:", pdb_file)
    return pdb_file


def convert_TPR(tpr, save_as="default", sel="protein", verbose=True):
    """
    Convert tpr for specific selection of atoms.
    Default: rex.tpr -> rex_protein.tpr

    Args:
        tpr (str):  <rex.tpr>
        save_as (str):
            "default": rex.tpr -> rex_protein.tpr
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "CA" or "calpha": CA atoms
            "BB" or "backbone": backbone atoms

    Returns:_
        tpr_file (str): realpath of converted tpr file

    Example:
        gmx code:
            echo 1|gmx convert-tpr -s rex.tpr -o rex_protein.tpr
        python code:
            convert_TPR(tpr="rex.tpr", save_as="rex_protein.tpr")
    """
    if sel.lower() == "system" or sel.lower() == "all":
        sel_code = 0
    elif sel.lower() == "protein":
        sel_code = 1
        if save_as == "default":
            pre = misc.get_base(tpr)
            save_as = f"{pre}_protein.tpr"
    elif sel.lower() == "ca" or sel.lower() == "calpha":
        sel_code = 3
    elif sel.lower() == "bb" or sel.lower() == "backbone":
        sel_code = 4
    else:
        raise ValueError("unsupported selection string.")

    misc.bash_cmd(f"echo {sel_code}|{gmx_cmd} convert-tpr -s {tpr} -o {save_as}")

    tpr_file = misc.realpath(save_as)
    if verbose:
        print("Saved converted tpr as:", tpr_file)
    return tpr_file


# def gmx_convert_TRP(s, o, sel="protein", **kwargs):
#     """
#     """
#     default = {"s": s,
#                "o": o,
#                "sel": sel}
#     cfg = misc.CONFIG(default, **kwargs)
#
#     if sel.lower() == "protein":
#         cfg.input = 1
#     gmx.convert_tpr_mpi(s=cfg.s, o=cfg.o, input=cfg.input)
#     return


def fix_XTC(tpr, xtc, save_as="default", pbc="mol", center=True, verbose=True):
    """
    Fix trajectory:
        - select only protein atoms
        - center molecule
        - (alternative: nojump)

    Args:
        tpr (str): <rex.tpr>
        xtc (str): <rex.xtc>
        save_as (str)
            "default": rex_protein.xtc
        pbc (str):
            "mol": puts the center of mass of molecules in the box
            "nojump": checks if atoms jump across the box and then puts them back
                      (i.e. all molecules will remain whole)
        center (bool): center the system in the box

    Returns:
        xtf_file (str): realpath of fixed xtc file (trajectory)

    Example:
        gmx code:
            echo 4 1|gmx trjconv -s rex.tpr -f traj.xtc -o traj_protein.xtc -ur compact -pbc mol -center
        python code:
            fix_XTC(tpr="rex.tpr", xtc="rex.xtc", save_as="rex_protein.xtc", center=True)
    """
    if save_as == "default":
        save_as = "rex_protein.xtc"
    if center:
        misc.bash_cmd(f"echo 1 1|{gmx_cmd} trjconv -s {tpr} -f {xtc} -o {save_as} -ur compact -pbc {pbc} -center")
    else:
        misc.bash_cmd(f"echo 1|{gmx_cmd} trjconv -s {tpr} -f {xtc} -o {save_as} -ur compact -pbc {pbc}")

    xtc_file = misc.realpath(save_as)
    if verbose:
        print("Saved fixed xtc as:", xtc_file)
    return xtc_file, 0


def get_RMSD(ref, xtc, save_as="default", tu="ns", verbose=True):
    """
    Calculate backbone RMSD.

    Args:
        ref (str): <protein.pdb>
        xtc (str): <rex_protein.xtc>
        save_as (str):
            "default": rmsd.xvg
        tu (str): time unit (ns, ps)

    Returns:
        RMSD_file (str): realpath to RMSD file

    Example:
        gmx code:
            echo 4 4|gmx rms -s <ref.pdb> -f rex_protein.xtc -o rmsd.xvg -tu ns
        python code:
            get_RMSD(ref="ref.pdb", xtc="rex_protein.xtc", save_as="rmsd.xvg", tu=ns)
    """
    if save_as == "default":
        save_as = "rmsd.xvg"
    misc.bash_cmd(f"echo 4 4|{gmx_cmd} rms -s {ref} -f {xtc} -o {save_as} -tu {tu}")

    RMSD_file = misc.realpath(save_as)
    if verbose:
        print("Saved RMSD file as:", RMSD_file)
    return RMSD_file
