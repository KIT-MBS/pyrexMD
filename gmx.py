import os
import glob
import gromacs
import MDAnalysis as mda
import myPKG.misc as _misc
import logging
logging.getLogger('gromacs.config').disabled = True
gromacs.environment.flags['capture_output'] = True  # print gromacs output

################################################################################
################################################################################


def __fix_ResourceWarning__():
    """
    Fix 'ResourceWarning: unclosed file' of GromacsWrapper's config.py module
    """
    module_path = f"{_misc.get_filedir(gromacs.__file__)}/config.py"
    old_str = "import sys"
    new_str = "import sys       ### fixed warning\n"
    new_str += "import warnings  ### fixed warning\n"
    new_str += "warnings.simplefilter('ignore', ResourceWarning)  ### fixed warning\n"

    needs_fix = True
    with open(module_path, "r") as f:
        old_text = f.read()
        if "### fixed warning" in old_text:
            needs_fix = False
        else:
            new_text = old_text.replace(old_str, new_str)
    if needs_fix:
        with open(module_path, "w") as f:
            f.write(new_text)
    return


### execute fix
__fix_ResourceWarning__()
################################################################################
################################################################################


def clean_up(path="./", pattern="gmx_pattern", verbose=True):
    """
    Clean up gromacs backup files with the patterns #*# and .*offsets.npz

    Note: pattern can be implemented into path variable directly.

    Args:
        path (str): directory path
        pattern (None/str): pattern
            (None): check for files with path only
            (str):  check for files with joined path of path + pattern
            "gmx_pattern": remove files with the patterns #*# and .*offsets.npz
        verbose (bool): print message ('removed file: ... ')
    """
    if pattern == "gmx_pattern":
        # iterate itself with gmx patterns
        clean_up(path=path, pattern="#*#", verbose=verbose)
        clean_up(path=path, pattern=".*offsets.npz", verbose=False)
        return

    elif pattern is None:
        realpath = os.path.realpath(path)
    else:
        realpath = _misc.joinpath(path, pattern)

    # remove files
    for item in glob.glob(realpath):
        os.remove(item)
        if verbose:
            print('removed file:', item)
    return


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


def pdb2gmx(f, o="protein.gro", odir="./", ff="amber99sb-ildn", water="tip3p", log=True, verbose=True, **kwargs):
    """
    Alias function of gromacs.pdb2gmx().

    Gromacs info:
        'pdb2gmx' reads a .pdb (or .gro) file, reads some database files, adds
        hydrogens to the molecules and generates coordinates in GROMACS (GROMOS),
        or optionally .pdb, format and a topology in GROMACS format. These files
        can subsequently be processed to generate a run input file.

    Args:
        f (str): input structure file: pdb gro tpr (g96 brk ent esp)
        o (str): output structure file: pdb gro (g96 brk ent esp)
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        ff (str): force field (see <gromacs_path>/top/<ff> for valid inputs)
            Protein: e.g.: amber99sb-ildn
            RNA: e.g.: amber14sb_OL15
        water (str): water model
        log (bool): save log file
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.pdb2gmx.help()
        # or terminal -> gmx pdb2gmx -h
        p (str): topology file: topol.top
        i (str): include file for topology: posre.itp
        n (str): index file: index.ndx

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.pdb2gmx(f=f, o=o, ff=ff, water=water.lower(), v=verbose)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/pdb2gmx.log"
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


# same function as TPR2PDB()
def editconf(f, o="default", odir="./", log=True, verbose=True, **kwargs):
    """
    Alias function of:
        TPR2PDB()
        gromacs.editconf()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro, or .g96.

    Args:
        f (str): tpr (gro g96 pdb brk ent esp)
        o (str): pdb (gro g96)
            "default": <f>.tpr -> <f>.pdb
                       <f>.gro -> box.gro
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        log (bool): save log file
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.editconf.help()
        # or terminal -> gmx editconf -h
        box (int): box vector lengths xyz (only 1 value for bt="cubic").
                   requires center=False to work.
        bt (str): box type: cubic triclinic dodecahedron octahedron
        c (bool): center molecule in box
        d (str): distance between the solute and the box

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if o == "default":
        if _misc.get_extension(f) == ".tpr":
            o = _misc.joinpath(odir, f"{_misc.get_base(f)}.pdb")
        elif _misc.get_extension(f) == ".gro":
            o = _misc.joinpath(odir, "box.gro")
    else:
        o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.editconf(f=f, o=o, **kwargs)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/editconf.log"
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


# same function as editconf()
def convert_TPR2PDB(tpr, o="default", odir="./", log=True, verbose=True, **kwargs):
    """
    Alias function of:
        editconf()
        gromacs.editconf()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro or .g96.

    Args:
        tpr (str): tpr (gro g96 pdb brk ent esp)
        o (str): pdb (gro g96)
            "default": <filename>.tpr -> <filename>.pdb
                       <filename>.gro -> box.gro
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        log (bool): save log file
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.editconf.help()
        # or terminal -> gmx editconf -h
        bt (str): box type: cubic triclinic dodecahedron octahedron
        c (bool): center molecule in box
        d (str): distance between the solute and the box

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if o == "default":
        if _misc.get_extension(tpr) == ".tpr":
            o = _misc.joinpath(odir, f"{_misc.get_base(tpr)}.pdb")
        elif _misc.get_extension(tpr) == ".gro":
            o = _misc.joinpath(odir, "box.gro")
    else:
        o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.editconf(f=tpr, o=o, **kwargs)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/editconf_TPR2PDB.log"
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


def convert_TPR(s, o="default", odir="./", sel="protein", verbose=True, **kwargs):
    """
    Alias function of gromacs.convert_tpr().

    Gromacs info:
        'gmx convert-tpr' can edit run input files in three ways:
        - modify tpr settings
        - create new tpr for a subset of original tpr
        - setting the charges of a specified group to zero(useful when doing
          free energy estimates using the LIE (Linear Interaction Energy) method.

    Args:
        s (str): tpr
        o (str): tpr
            "default": <s>.tpr -> <s>_<sel>.tpr
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        sel (str): selection string(case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.convert_tpr.help()
        # or terminal -> gmx convert_tpr -h

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    sel_code = _get_sel_code(sel)
    if o == "default":
        o = _misc.joinoath(odir, f"{_misc.get_base(s)}_{sel.lower()}.tpr")
    else:
        o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        gromacs.convert_tpr(s=s, o=o, input=sel_code, **kwargs)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint("Saved file as: {o_file}", cfg.cprint_color)

    #clean up
    clean_up(path=_misc.dirpath(o_file), pattern=".*offsets.npz", verbose=False)
    clean_up(path=_misc.dirpath(o_file), pattern="#*_protein.tpr*#", verbose=False)
    return o_file


def grompp(f, o, c, p="topol.top", log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    Alias function of gromacs.grompp().

    Args:
        f (str): input file: mdp
        o (str): output file: tpr
        c (str): structure file: gro tpr pdb (g96 brk ent esp)
        p (str): topology file: top
        log (bool): save log file
        log_overwrite (bool):
            True: save log file as <logs/gromp.log>
            False: save log file as <logs/gromp_{i}.log> for i=1,...,200
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.grompp.help()
        # or terminal -> gmx grompp -h
        maxwarn (int): Number of allowed warnings during input processing.
                       Not for normal use and may generate unstable systems.

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"maxwarn": 10,
               "cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    odir = _misc.realpath(_misc.dirpath(o))
    o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.grompp(f=f, o=o, c=c, p=p, v=verbose,
                                               maxwarn=cfg.maxwarn, **kwargs)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/grompp.log"
        if not log_overwrite:
            log_names = [f"{logdir}/grompp_{i}.log" for i in range(1, 201)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def solvate(cp, cs="spc216.gro", o="solvent.gro", p="topol.top",
            log=True, verbose=True, **kwargs):
    """
    Alias function of gromacs.solvate().

    cp is usually "box.gro"

    Args:
        cp (str): structure file ~ solute:  gro pdb tpr (g96 brk ent esp)
        cs (str): structure file ~ solvent: gro pdb tpr (g96 brk ent esp)
            Note: "spc216.gro" is used for all 3-point water models
        o (str): gro pdb (g96 brk ent esp)
            default case: save in same directory as cp
            special case: if o is relative/absolute path -> save there
        p (str): topology file: topol.top
        log (bool): save log file
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.solvate.help()
        # or terminal -> gmx solvate -h
        maxsol (int): maximum number of solvent molecules to add if they fit in
                      the box.
            0 (default): ignore this setting

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    odir = _misc.realpath(_misc.dirpath(cp))
    o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.solvate(cp=cp, cs=cs, o=o, p=p, **kwargs)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/solvate.log"
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def genion(s, o, p="topol.top", input="13", pname="NA", nname="CL",
           neutral=True, log=True, verbose=True, **kwargs):
    """
    Alias fuction of gromacs.genion().

    Note: input 13 (=SOL group) seems to be the only selection group that works
          with genion.

    Gromacs info:
        gmx genion randomly replaces solvent molecules with monoatomic ions.
        The group of solvent molecules should be continuous and all molecules
        should have the same number of atoms.

    Args:
        s (str): structure file: tpr
        o (str): output file: gro pdb (g96 brk ent esp)
        p (str): topology file: topol.top
        input (str): selection grp
            default: 13 (SOL)
        pname (str): positive ion name
        nname (str): negative ion name
        neutral (bool): add enough ions to neutralize the system. These ions are
                        added on top of those specified with -np/-nn or -conc.
        log (bool): save log file
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.genion.help()
        # or terminal -> gmx genion -h

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    odir = _misc.realpath(_misc.dirpath(o))
    o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.genion(s=s, o=o, p=p, pname=pname,
                                               nname=nname, neutral=neutral,
                                               input=str(input), **kwargs)
        print(stderr)
        print()
        print(stdout)

    # clean up
    odir = _misc.dirpath(_misc.realpath(o))
    clean_up("./", verbose=False)
    clean_up(odir, verbose=False)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    if log:
        logdir = _misc.mkdir(f"{odir}/logs")
        logfile = f"{logdir}/genion.log"
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved  log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def mdrun(verbose=True, **kwargs):
    """
    Alias fuction of gromacs.mdrun().

    Note:
        output can be printed only after mdrun has finished.
        To see realtime updates invoke the command using "!gmx_mpi mdrun <parameters>"

    Args:
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.mdrun.help()
        # or terminal -> gmx mdrun -h
        deffnm (str): default filename
        nsteps (int): maximum number of steps (used instead of mdp setting)
    """
    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.mdrun(v=True, **kwargs)

    return


def trjconv(s, f, o="default", odir="./", sel="protein", verbose=True, **kwargs):
    """
    Alias function of gromacs.trjconv().

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
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.trjconv.help()
        # or terminal -> gmx trjconf -h

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
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
        o = _misc.joinpath(odir, f"{_misc.get_base(f)}_{sel.lower()}.xtc")
    else:
        o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        gromacs.trjconv(s=s, f=f, o=o, input=sel_code, **kwargs)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    #clean up
    clean_up(path=_misc.dirpath(o_file), pattern=".*offsets.npz", verbose=False)
    clean_up(path=_misc.dirpath(o_file), pattern="#*_protein.xtc*#", verbose=False)
    return o_file


def fix_TRAJ(tpr, xtc, o="default", odir="./", tu="ns", sel="protein", pbc="mol", center=True,
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
                os (str): filename of new structure (selection only)
                of (str): filename of new trajectory (selection only)
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        tu (str): time unit: fs ps ns us ms s
        sel (str): selection string(case independant)
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
        # see python  -> gromacs.trjconv.help()
        # or terminal -> gmx trjconv -h

    Returns:
        tpr_file (str): realpath of new tpr file (selection only)
        xtc_file (str): realpath of new xtc file (selection only)
    """
    # check if o arg is "default" or list
    if o == "default":
        o = [_misc.joinpath(odir, f"{_misc.get_base(tpr)}_{sel.lower()}.tpr"),
             _misc.joinpath(odir, f"{_misc.get_base(xtc)}_{sel.lower()}.xtc")]
    else:
        # error test
        if not isinstance(o, list):
            raise ValueError("""Invalid value of 'o' argument.
    o (str or list): filenames of new structure and new trajectory (both selection only)
        "default" (str):
            <tpr>.tpr -> <xtc>_<sel>.tpr
            <xtc>.xtc -> <xtc>_<sel>.xtc
        [os, of] (list):
            os (str): filename of new structure (selection only)
            of (str): filename of new trajectory (selection only)""")
        # if no error: joinpath
        o = [_misc.joinpath(odir, o[0]),
             _misc.joinpath(odir, o[1])]

    ### GromacsWrapper
    print("Fixing topology:")
    tpr_file = convert_TPR(s=tpr, o=o[0], odir=odir, sel=sel, verbose=verbose)  # selection only

    print("Fixing trajectory:")
    xtc_file = trjconv(s=tpr_file, f=xtc, o=o[1], odir=odir, tu=tu, sel=sel, pbc=pbc,
                       center=center, verbose=verbose, **kwargs)

    #clean up
    clean_up(path=_misc.dirpath(tpr_file), verbose=verbose)
    clean_up(path=_misc.dirpath(xtc_file), verbose=verbose)
    return tpr_file, xtc_file


def get_RMSD(ref, xtc, o="default", odir="./", tu="ns", sel="bb", verbose=True, **kwargs):
    """
    Alias function of gromacs.rms().
    Calculate backbone RMSD.

    Args:
        ref (str): reference structure: pdb (tpr gro g96 brk ent)
        xtc (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str): xvg
            "default": rmsd.xvg
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        tu (str): time unit: fs ps ns us ms s
        sel (str): selection string (case independant)
            "system" or "all": all atoms
            "protein": protein atoms
            "ca" or "calpha": CA atoms
            "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.rms.help()
        # or terminal -> gmx rms -h

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # check input files for same atom count
    u1 = mda.Universe(ref)
    u2 = mda.Universe(xtc)
    if u1.atoms.n_atoms != u2.atoms.n_atoms:
        raise ValueError(f"""ref and xtc file have not the same atom count!

        Atoms in ref: {u1.atoms.n_atoms}
        Atoms in xtc: {u2.atoms.n_atoms}""")

    if o == "default":
        o = _misc.joinpath(odir, "rmsd.xvg")
    else:
        o = _misc.joinpath(odir, o)

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        gromacs.rms(s=ref, f=xtc, o=o, tu=tu, **kwargs)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)
    return o_file


def get_ref_structure(f, o="default", odir="./", ff="amber99sb-ildn", water="tip3p",
                      verbose=True, **kwargs):
    """
    Creates ref structure of input structure via pdb2gmx to fix atom count by
    applying force field.

    Args:
        f (str): input structure file: pdb gro tpr (g96 brk ent esp)
        o (str): output structure file: pdb gro trp (g96 brk ent esp)
        odir (str): output directory
            special case: odir is ignored when o is relative/absolute path
        ff (str): force field (see <gromacs_path>/top/<ff> for valid inputs)
            Protein: e.g.: amber99sb-ildn
            RNA: e.g.: amber14sb_OL15
        water (str): water model
        verbose (bool): print/mute gromacs messages

    Kwargs:
        # see python  -> gromacs.pdb2gmx.help()
        # or terminal -> gmx pdb2gmx -h

        cprint_color (str)

    Returns:
        o_file (str): realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if o == "default":
        o = _misc.joinpath(odir, f"{_misc.get_base(f)}_ref.pdb")
    else:
        o = _misc.joinpath(odir, o)

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        gromacs.pdb2gmx(f=f, o=o, ff=ff, water=water.lower(), **kwargs)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)
    return o_file


################################################################################
################################################################################
### DCA REX setup functions
### -> allow different start conformations with fixed boxsize and solution number
def WF_getParameter_boxsize(logfile="./logs/editconf.log", base=0.2, verbose=True):
    """
    Read <logfile> and suggest a 'fixed boxsize' parameter for REX simulations.

    Args:
        logfile (str): path to <editconf.log> containing the line 'system size: X Y Z'
        base (float): round up base for highest box dimension taken from <logfile>
        verbose (bool)

    Returns:
        boxsize (float): suggested boxsize parameter
    """
    boxsize = None

    with _misc.HiddenPrints(verbose=verbose):
        with open(logfile, "r") as fin:
            _misc.cprint(f"Reading logfile: {logfile}")
            for line in fin:

                if "new box vectors" in line:
                    s = line.split()
                    dims = [float(x) for x in s if x.replace(".", "", 1).isdigit()]
                    boxsize = round(_misc.round_up(max(dims), base=base), 2)
                    _misc.cprint(line, "blue", end='')

                elif "system size" in line or "diameter" in line or "box volume" in line:
                    _misc.cprint(line, "blue", end='')

        if boxsize is None:
            raise _misc.Error("No boxsize parameters found.")
        else:
            _misc.cprint(f"suggested box size: {boxsize}", "green", end='')
    return boxsize


def WF_getParameter_maxsol(logfile="./logs/solution.log", maxsol_reduce=50, verbose=True):
    """
    Read <logfile> and suggest a 'max solution' parameter for REX simulations.

    Args:
        logfile (str): path to <editconf.log> containing the line 'system size: X Y Z'
        maxsol_reduce (int): reduce max solution number taken from <logfile>
                          -> guarantees fixed solution number for different
                             rstart configurations
        verbose (bool)

    Returns:
        maxsol (int): suggested max solution parameter
    """
    with open(logfile, "r") as fin:
        for line in fin:
            if "Number of solvent molecules" in line:
                s = line.split()
                maxsol = [int(x)-maxsol_reduce for x in s if x.isdigit()][0]

                if verbose:
                    _misc.cprint(f"Reading logfile: {logfile}")
                    _misc.cprint(line, "blue", end="")
                    _misc.cprint(f"suggested max solution: {maxsol}", "green", end="")
                return maxsol

    _misc.Error("No solution parameter found.")
    return


def WF_REX_setup(rex_dirs, boxsize, maxsol, verbose=False, verbose_gmx=False):
    """
    Workflow: REX setup (without energy minimization)

    Args:
        rex_dirs (list): list with rex_dirs (output of rex.get_REX_DIRS())
        boxsize (float): suggested boxsize parameter (output of gmx.WF_getParameter_boxsize())
        maxsol (int): suggested max solution parameter (output of gmx.WF_getParameter_max_solution())
        verbose (bool): show blue log messages (saved file as, saved log as)
        verbose_gmx (bool): show gmx module messages (requires verbose=True)
    """
    for rex_dir in rex_dirs:
        _misc.cprint("#######################################################################################")
        _misc.cd(rex_dir)
        decoy_pdb = _misc.get_filename("*_ref.pdb")
        _misc.cprint(f"Using decoy pdb: {decoy_pdb}")

        # 1) generate topology
        _misc.cprint(f"\nGenerating topology...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            protein_gro = pdb2gmx(f=decoy_pdb, verbose=False)

        # 2) generate box
        _misc.cprint(f"Generating box with fixed size ({boxsize}) ...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            editconf(f=protein_gro, o="box.gro", bt="cubic", box=boxsize, c=True, verbose=verbose_gmx)

        # 3) generate solvent
        _misc.cprint(f"Generating solvent with fixed solvent molecules ({maxsol})...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            solvate(cp="box.gro", maxsol=maxsol, verbose=verbose_gmx)

        # 4) generate ions
        _misc.cprint(f"Generating ions...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            grompp(f="../ions.mdp", o="ions.tpr", c="solvent.gro", p="topol.top", verbose=verbose_gmx)
            genion(s="ions.tpr", o="ions.gro", p="topol.top", verbose=False)

    _misc.cprint("#######################################################################################")
    _misc.cd("..")
    _misc.cprint("Finished setup of all REX DIRS (skipped energy minimization).", "green")
    return


def WF_REX_setup_energy_minimization(rex_dirs, nsteps=None, verbose=False):
    """
    Workflow: REX energy minimization

    Args:
        rex_dirs (list): list with rex_dirs (output of rex.get_REX_DIRS())
        nsteps (None/int): maximum number of steps
            None: use .mdp option
            int: use instead of .mdp option
        verbose (bool): show/hide gromacs output
    """
    for rex_dir in rex_dirs:
        _misc.cprint("#######################################################################################")
        _misc.cd(rex_dir)

        # 5) energy minimization
        _misc.cprint(f"Performing energy minimization...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            _ = grompp(f="../min.mdp", o="em.tpr", c="ions.gro", p="topol.top")
            if nsteps is None:
                mdrun(deffnm="em", verbose=verbose)
            else:
                mdrun(deffnm="em", nsteps=nsteps, verbose=verbose)
            clean_up("./", verbose=False)

    _misc.cprint("#######################################################################################")
    _misc.cd("..")
    _misc.cprint("Finished energy minimization of all REX DIRS.", "green")
################################################################################
################################################################################
### TEMPLATES


def get_template_1_EM(gmx_mpi=True):
    """
    Args:
        gmx_mpi (bool): use gmx_mpi instead of gmx
    """
    print("Energy Minimization Template:")
    if gmx_mpi:
        template = "!gmx_mpi grompp -f min.mdp -o em.tpr -c ions.gro -p topol.top\n"
        template += "!gmx_mpi mdrun -v -deffnm em"
    else:
        template = "!gmx grompp -f min.mdp -o em.tpr -c ions.gro -p topol.top\n"
        template += "!gmx mdrun -v -deffnm em"
    print(template)
    return


def get_template_2_NVT(gmx_mpi=True):
    """
    Args:
        gmx_mpi (bool): use gmx_mpi instead of gmx
    """
    print("NVT Template:")
    if gmx_mpi:
        template = "!gmx_mpi grompp -f nvt.mdp -o nvt.tpr -c em.gro -r em.gro -p topol.top\n"
        template += "!gmx_mpi mdrun -v -deffnm nvt"
    else:
        template = "!gmx grompp -f nvt.mdp -o nvt.tpr -c em.gro -r em.gro -p topol.top\n"
        template += "!gmx mdrun -v -deffnm nvt"
    print(template)
    return


def get_template_3_NPT(gmx_mpi=True):
    """
    Args:
        gmx_mpi (bool): use gmx_mpi instead of gmx
    """
    print("NPT Template:")
    if gmx_mpi:
        template = "!gmx_mpi grompp -f npt.mdp -o npt.tpr -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top\n"
        template += "!gmx_mpi mdrun -v -deffnm npt"
    else:
        template = "!gmx grompp -f npt.mdp -o npt.tpr -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top\n"
        template += "!gmx mdrun -v -deffnm npt"
    print(template)
    return


def get_template_4_MD(gmx_mpi=True):
    """
    Args:
        gmx_mpi (bool): use gmx_mpi instead of gmx
    """
    print("MD Template:")
    if gmx_mpi:
        template = "!gmx_mpi grompp -f md.mdp -o traj.tpr -c npt.gro -t npt.cpt -p topol.top\n"
        template += "!gmx_mpi mdrun -v -deffnm traj\n\n"
        template += "# with tabulated bonds add: -tableb table_b0.xvg\n"
        template += "# to append add: -cpi -append"
    else:
        template = "!gmx grompp -f md.mdp -o traj.tpr -c npt.gro -t npt.cpt -p topol.top\n"
        template += "!gmx mdrun -v -deffnm traj\n"
        template += "# with tabulated bonds add: -tableb table_b0.xvg\n"
        template += "# to append add: -cpi -append"
    print(template)
    return
