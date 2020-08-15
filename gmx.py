import gromacs
import MDAnalysis as mda
import myPKG.misc as misc
from myPKG.misc import HiddenPrints


def _get_sel_code(sel, **kwargs):
    """
    Get selection code.

    Args:
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
    Returns:
        sel_code (str)
            "0": code for system
            "1": code for protein
            "3": code for CA
            "4": code for BB
    """
    if sel.lower() == "system" or sel.lower() == "all":
        sel_code = "0"
    elif sel.lower() == "protein":
        sel_code = "1"
    elif sel.lower() == "ca" or sel.lower() == "calpha":
        sel_code = "3"
    elif sel.lower() == "bb" or sel.lower() == "backbone":
        sel_code = "4"
    else:
        raise ValueError("Unsupported selection string. Modify code to add more cases")

    return sel_code


# same function as TPR2PDB()
def editconf(f, o="default", verbose=True, **kwargs):
    """
    Alias function to
        TPR2PDB()
        gromacs.editconf()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro, or .g96.

    Args:
        f (str): tpr (gro g96 pdb brk ent esp)
        o (str): pdb (gro g96)
            "default": <f>.tpr -> <f>.pdb
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.editconf.help()

    Returns:
        o_file (str): realpath of output file
    """
    if o == "default":
        o = f"{misc.get_base(f)}.pdb"

    # GromacsWrapper
    if verbose:
        gromacs.editconf(f=f, o=o, **kwargs)
    else:
        with HiddenPrints():
            gromacs.editconf(f=f, o=o, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file


# same function as editconf()
def convert_TPR2PDB(tpr, o="default", verbose=True, **kwargs):
    """
    Alias function to
        editconf()
        gromacs.editconf()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro or .g96.

    Args:
        tpr (str): tpr (gro g96 pdb brk ent esp)
        o (str): pdb (gro g96)
            "default": <filename>.tpr -> <filename>.pdb
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.editconf.help()

    Returns:
        o_file (str): realpath of output file
    """
    if o == "default":
        o = f"{misc.get_base(tpr)}.pdb"

    # GromacsWrapper
    if verbose:
        gromacs.editconf(f=tpr, o=o, **kwargs)
    else:
        with HiddenPrints():
            gromacs.editconf(f=tpr, o=o, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file


def convert_TPR(s, o="default", sel="protein", verbose=True, **kwargs):
    """
    Alias function gromacs.convert_tpr().

    Gromacs info:
        'gmx convert-tpr' can edit run input files in three ways:
        - modify tpr settings
        - create new tpr for a subset of original tpr
        - setting the charges of a specified group to zero (useful when doing
          free energy estimates using the LIE (Linear Interaction Energy) method.

    Args:
        s (str): tpr
        o (str): tpr
            "default": <s>.tpr -> <s>_<sel>.tpr
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.convert_tpr.help()

    Returns:
        o_file (str): realpath of output file
    """
    sel_code = _get_sel_code(sel)
    if o == "default":
        o = f"{misc.get_base(s)}_{sel.lower()}.tpr"

    # GromacsWrapper
    if verbose:
        gromacs.convert_tpr(s=s, o=o, input=sel_code, **kwargs)
    else:
        with HiddenPrints():
            gromacs.convert_tpr(s=s, o=o, input=sel_code, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file


def trjconv(s, f, o="default", sel="protein", verbose=True, **kwargs):
    """
    Alias function to gromacs.trjconv().

    Gromacs info:
        gmx trjconv can convert trajectory files in many ways

        - from one format to another
        - select a subset of atoms
        - center atoms in the box
        - fit atoms to reference structure
        - reduce the number of frames
        ... etc.

    Args:
        s (str): structure: tpr (gro g96 pdb brk ent)
        f (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str): trajectory: xtc (trr gro g96 pdb tng)
            "default": <f>.xtc -> <f>_<sel>.xtc
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.trjconv.help()

    Returns:
        o_file (str): realpath of output file
    """
    # check input files for same atom count
    u1 = mda.Universe(s)
    u2 = mda.Universe(f)
    if u1.atoms.n_atoms != u2.atoms.n_atoms:
        raise ValueError(f"""s and f file have not the same atom count!

        Atoms in s: {u1.atoms.n_atoms}
        Atoms in f: {u2.atoms.n_atoms}""")

    sel_code = _get_sel_code(sel)
    if "center" in kwargs:
        if kwargs["center"] is True:
            sel_code = f"{sel_code} {sel_code}"
    if o == "default":
        o = f"{misc.get_base(f)}_{sel.lower()}.xtc"

    # GromacsWrapper
    if verbose:
        gromacs.trjconv(s=s, f=f, o=o, input=sel_code, **kwargs)
    else:
        with HiddenPrints():
            gromacs.trjconv(s=s, f=f, o=o, input=sel_code, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file


def fix_TRAJ(tpr, xtc, o="default", tu="ns", sel="protein", pbc="mol", center=True,
             verbose=True, **kwargs):
    """
    Fix topology:
        - convert tpr to selection only
        - if o="default": save as <s>_<sel>.tpr

    Fix trajectory:
        - selection only
        - tu ns
        - pbc mol
        - center
        - if o="default": save as <f>_<sel>.xtc

    Args:
        tpr (str): structure: tpr
        xtc (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str or list): filenames of new structure and new trajectory (both selection only)
            "default" (str):
                <tpr>.tpr -> <xtc>_<sel>.tpr
                <xtc>.xtc -> <xtc>_<sel>.xtc
            [os, of] (list):
                os (str):  filename of new structure (selection only)
                of (str):  filename of new trajectory (selection only)
        tu (str): time unit: fs ps ns us ms s
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        pbc (str): PBC treatment: none mol res atom nojump cluster whole
            "mol": puts the center of mass of molecules in the box.
                   requires structure file s.
            "nojump": checks if atoms jump across the box and then puts them back
                      (i.e. all molecules will remain whole)
            (see gromacs help text for other descriptions)
        center (bool): center atoms in box
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.trjconv.help()

    Returns:
        tpr_file (str): realpath of new tpr file (selection only)
        xtc_file (str): realpath of new xtc file (selection only)
    """
    # check if o arg is "default" or list
    if o == "default":
        o = [f"{misc.get_base(tpr)}_{sel.lower()}.tpr",
             f"{misc.get_base(xtc)}_{sel.lower()}.xtc"]
    else:
        if not isinstance(o, list):
            raise ValueError("""Invalid value of 'o' argument.
    o (str or list): filenames of new structure and new trajectory (both selection only)
        "default" (str):
            <s>.tpr -> <s>_<sel>.tpr
            <f>.xtc -> <f>_<sel>.xtc
        [os, of] (list):
            os (str):  filename of new structure (selection only)
            of (str):  filename of new trajectory (selection only)""")

    ### GromacsWrapper
    # fix topology
    tpr_file = convert_TPR(s=tpr, o=o[0], sel=sel, verbose=verbose)  # selection only
    # fix trajectory
    xtc_file = trjconv(s=tpr_file, f=xtc, o=o[1], tu=tu, sel=sel, pbc=pbc,
                       center=center, verbose=verbose, **kwargs)
    return tpr_file, xtc_file


def get_RMSD(ref, xtc, o="default", tu="ns", sel="bb", verbose=True, **kwargs):
    """
    Alias function to gromacs.rms().
    Calculate backbone RMSD.

    Args:
        ref (str): reference structure: pdb (tpr gro g96 brk ent)
        xtc (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str): xvg
            "default": rmsd.xvg
        tu (str): time unit: fs ps ns us ms s
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.rms.help()

    Returns:
        o_file (str): realpath of output file
    """
    # check input files for same atom count
    u1 = mda.Universe(ref)
    u2 = mda.Universe(xtc)
    if u1.atoms.n_atoms != u2.atoms.n_atoms:
        raise ValueError(f"""ref and xtc file have not the same atom count!

        Atoms in ref: {u1.atoms.n_atoms}
        Atoms in xtc: {u2.atoms.n_atoms}""")

    if o == "default":
        o = "rmsd.xvg"

    # GromacsWrapper
    if verbose:
        gromacs.rms(s=ref, f=xtc, o=o, tu=tu, **kwargs)
    else:
        with HiddenPrints():
            gromacs.rms(s=ref, f=xtc, o=o, tu=tu, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file


def get_ref_structure(f, o="default", water="tip3p", input="6", verbose=True, **kwargs):
    """
    Creates ref structure of input structure via pdb2gmx to fix atom count.

    Args:
        f (str):  input structure file: pdb gro tpr (g96 brk ent esp)
        o (str): output structure file: pdb gro trp (g96 brk ent esp)
        water (str): water model
        input (str): force field number
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see gromacs.pdb2gmx.help()

    Returns:
        o_file (str): realpath of output file
    """
    if o == "default":
        o = f"{misc.get_base(f)}_ref.pdb"

    # GromacsWrapper
    if verbose:
        gromacs.pdb2gmx(f=f, o=o, water=water, input=input, **kwargs)
    else:
        with HiddenPrints():
            gromacs.pdb2gmx(f=f, o=o, water=water, input=input, **kwargs)

    # save message
    o_file = misc.realpath(o)
    print("Saved file as:", o_file)
    return o_file

################################################################################
### WORKFLOW FUNCTIONS


def WF_get_RMSD(cfg, verbose=True, **kwargs):
    """
    1) convert TPR:
        - select protein atoms
        - save as <filename>_protein.tpr
    2) fix trajectory:
        - select protein atoms
        - tu ns
        - pbc mol
        - center
        - save as <filename>_protein.xtc
    3) get backbone RMSD:
        - tu ns
        - save as rmsd.xvg

    Args:
        files_cfg (misc.CONFIG): config class with (real)paths to input files
            KEYWORDS:  ARGUMENTS
            ref (str): (real)path to reference pdb (protein atoms)
            tpr (str): (real)path to .trp file (all atoms/protein atoms)
            xtc (str): (real)path to .xtc file (protein atoms)
        verbose (bool): print/mute gromacs messages

    Kwargs:
        see KWARGS from gmx.fix_TRAJ() and gmx.get_RMSD().

    Returns:
        cfg (misc.CONFIG): config class with (real)paths to input and output files
            KEYWORDS:  ARGUMENTS
            ref (str): (real)path to reference pdb (protein atoms)
            tpr (str): (real)path to .trp file (all atoms/protein atoms)
            tpr_protein (str): (real)path to .trp file (all atoms/protein atoms)
            xtc (str): (real)path to .xtc file (protein atoms)
            xtc_protein (str): realpath to fixed trajectory file
            rmsd_file (str): realpath to RMSD file
    """
    cfg = misc.CONFIG(cfg, **kwargs)

    ## TODO

    # fix trajectory
    #tpr_protein, xtc_protein = fix_TRAJ(s=cfg.tpr_protein, f=cfg.xtc, **kwargs)
    #cfg.update_config(tpr_protein=tpr_protein, xtc_protein=xtc_protein)

    # get backbone RMSD
    #rmsd_file = get_RMSD(ref=cfg.ref, xtc=cfg.xtc_protein, verbose=verbose)
    #cfg.update_config(rmsd_file=rmsd_file)
    return cfg
