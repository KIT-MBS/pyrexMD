# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: gmx.py
# @Last modified by:   arthur
# @Last modified time: 17.05.2021

"""
This module contains functions to interact with `GROMACS` to setup systems and
run MD simulations.


Example:
--------

.. code-block:: python

    import pyrexMD.gmx as gmx
    
    # create ref pdb:
    pdb = "./1l2y.pdb"
    ref = gmx.get_ref_structure(pdb, ff='amber99sb-ildn', water='tip3p', ignh=True)

    # generate topology & box
    gmx.pdb2gmx(f=ref, o="protein.gro", ff='amber99sb-ildn', water='tip3p', ignh=True)
    gmx.editconf(f="protein.gro", o="box.gro", d=2.0, c=True, bt="cubic")

    # generate solvent & ions
    gmx.solvate(cp="box.gro", o="solvent.gro")
    gmx.grompp(f="ions.mdp", o="ions.tpr",c="solvent.gro")
    gmx.genion(s="ions.tpr", o="ions.gro", neutral=True, input="SOL")

    # minimize
    gmx.grompp(f="min.mdp", o="min.tpr", c="ions.gro")
    gmx.mdrun(deffnm="min")

    # NVT equilibration
    gmx.grompp(f="nvt.mdp", o="nvt.tpr", c="min.gro", r="min.gro")
    gmx.mdrun(deffnm="nvt")

    # NPT equilibration
    gmx.grompp(f="npt.mdp", o="npt.tpr", c="nvt.gro", r="nvt.gro", t="nvt.cpt")
    gmx.mdrun(deffnm="npt")

    # MD run
    gmx.grompp(-f"md.mdp", o="traj.tpr", c="npt.gro", t="npt.cpt")
    gmx.mdrun(deffn="traj")
"""

import os
import glob
import gromacs
import MDAnalysis as mda
import pyrexMD.misc as _misc
import numpy as np
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


def clean_up(path="./", pattern="gmx_pattern", ignore=None, verbose=True):
    """
    Clean up gromacs backup files with the patterns #*# and .*offsets.npz

    .. Note:: pattern can be implemented into path variable directly.

    Args:
        path (str): directory path
        pattern (None, str):
          | pattern
          | None: check for files with path only
          | str:  check for files with joined path of path + pattern
          | "gmx_pattern": remove files with the patterns #*# and .*offsets.npz
        ignore (None, str, list of str, tuple of str):
          | one or multiple ignore pattern
          | None: no ignore pattern
          | str: single ignore pattern
          | list of str, tuple of str: list/tuple with multiple ignore patterns
        verbose (bool): print message ('removed file: ... ')
    """
    if pattern == "gmx_pattern":
        # iterate itself with gmx patterns
        clean_up(path=path, pattern="#*#", ignore=ignore, verbose=verbose)
        clean_up(path=path, pattern=".*offsets.npz", ignore=ignore, verbose=False)
        return

    elif pattern is None:
        realpath = os.path.realpath(path)
    else:
        realpath = _misc.joinpath(path, pattern)

    # remove files
    for item in glob.glob(realpath):
        if isinstance(ignore, str):
            if ignore in item:
                continue
        if isinstance(ignore, (list, tuple)):
            for ign in ignore:
                if ign in item:
                    continue
        os.remove(item)
        if verbose:
            print('removed file:', item)
    return


def _get_sel_code(sel, **kwargs):
    """
    Get selection code.

    Args:
        sel (str):
          | selection string (case independant)
          | "system" or "all": all atoms
          | "protein": protein atoms
          | "ca" or "calpha": CA atoms
          | "bb" or "backbone": backbone atoms
    Returns:
        sel_code (str)
          | "0": code for system
          | "1": code for protein
          | "3": code for CA
          | "4": code for BB
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


def pdb2gmx(f, o="protein.gro", odir="./", ff="amber99sb-ildn", water="tip3p",
            ignh=True, log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    Modified function of gromacs.pdb2gmx().

    Gromacs info:
        'pdb2gmx' reads a .pdb (or .gro) file, reads some database files, adds
        hydrogens to the molecules and generates coordinates in GROMACS (GROMOS),
        or optionally .pdb, format and a topology in GROMACS format. These files
        can subsequently be processed to generate a run input file.

    Args:
        f (str): input structure file: pdb gro tpr (g96 brk ent esp)
        o (str): output structure file: pdb gro (g96 brk ent esp)
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        ff (str):
          | force field (see <gromacs_path>/top/<ff> for valid inputs)
          | "amber99sb-ildn" etc. for proteins
          | "amber14sb_OL15" etc. for RNAs
        water (str): water model
        ignh (bool): ignore hydrogen
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/pdb2gmx.log>
          | False: save log file as <logs/pdb2gmx_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        p (str): topology file: topol.top
        i (str): include file for topology: posre.itp
        n (str): index file: index.ndx
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.pdb2gmx.help()
        - terminal -> gmx pdb2gmx -h

    Returns:
        o_file (str)
            realpath of output file
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    o = _misc.joinpath(odir, o)  # special case for joinpath

    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.pdb2gmx(f=f, o=o, ff=ff, water=water.lower(), ignh=ignh, v=verbose)
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
        if not log_overwrite:
            log_names = [f"{logdir}/pdb2gmx_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


# same function as TPR2PDB()
def editconf(f, o="default", odir="./", log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    - Modified function of gromacs.editconf()
    - Alias function of TPR2PDB()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro, or .g96.

    Args:
        f (str): tpr (gro g96 pdb brk ent esp)
        o (str):
          | pdb (gro g96)
          | "default": <f>.tpr -> <f>.pdb
          | <f>.gro -> box.gro
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/editconf.log>
          | False: save log file as <logs/editconf_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        box (int):
          | box vector lengths xyz (only 1 value for bt="cubic").
          | requires center=False to work.
        bt (str): box type: cubic triclinic dodecahedron octahedron
        c (bool): center molecule in box
        d (str): distance between the solute and the box
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python  -> gromacs.editconf.help()
        - terminal -> gmx editconf -h

    Returns:
        o_file (str)
            realpath of output file
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
        if not log_overwrite:
            log_names = [f"{logdir}/editconf_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


# same function as editconf()
def convert_TPR2PDB(tpr, o="default", odir="./", log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    - Modified function of gromacs.editconf()
    - Alias function of editconf()

    Gromacs info:
        'gmx editconf' converts generic structure format to .pdb, .gro or .g96.

    Args:
        tpr (str): tpr (gro g96 pdb brk ent esp)
        o (str):
          | pdb (gro g96)
          | "default": <filename>.tpr -> <filename>.pdb
          |            <filename>.gro -> box.gro
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/editconf.log>
          | False: save log file as <logs/editconf_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        bt (str): box type: cubic triclinic dodecahedron octahedron
        c (bool): center molecule in box
        d (str): distance between the solute and the box

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.editconf.help()
        - terminal -> gmx editconf -h

    Returns:
        o_file (str)
            realpath of output file
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
        if not log_overwrite:
            log_names = [f"{logdir}/editconf_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)

    return o_file


def convert_TPR(s, o="default", odir="./", sel="protein", verbose=True, **kwargs):
    """
    Modified function of gromacs.convert_tpr().

    Gromacs info:
        'gmx convert-tpr' can edit run input files in three ways:
        - modify tpr settings
        - create new tpr for a subset of original tpr
        - setting the charges of a specified group to zero(useful when doing free energy estimates using the LIE (Linear Interaction Energy) method.

    Args:
        s (str): tpr
        o (str):
          | tpr
          | "default": <s>.tpr -> <s>_<sel>.tpr
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        sel (str):
          | selection string(case independant)
          | "system" or "all": all atoms
          | "protein": protein atoms
          | "ca" or "calpha": CA atoms
          | "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.convert_tpr.help()
        - terminal -> gmx convert_tpr -h

    Returns:
        o_file (str)
            realpath of output file
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
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)

    #clean up
    clean_up(path=_misc.dirpath(o_file), pattern=".*offsets.npz", verbose=False)
    clean_up(path=_misc.dirpath(o_file), pattern="#*_protein.tpr*#", verbose=False)
    return o_file


def grompp(f, o, c, p="topol.top", log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    Modified function of gromacs.grompp().

    Args:
        f (str): input file: mdp
        o (str): output file: tpr
        c (str): structure file: gro tpr pdb (g96 brk ent esp)
        p (str): topology file: top
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/grompp.log>
          | False: save log file as <logs/grompp_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        maxwarn (int):
          | Number of allowed warnings during input processing.
          | Not for normal use and may generate unstable systems.
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.grompp.help()
        - terminal -> gmx grompp -h

    Returns:
        o_file (str)
            realpath of output file
    """
    default = {"maxwarn": 10,
               "cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    if "maxwarn" in kwargs:
        del kwargs["maxwarn"]
    if "cprint_color" in kwargs:
        del kwargs["cprint_color"]
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
            log_names = [f"{logdir}/grompp_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def solvate(cp, cs="spc216.gro", o="solvent.gro", p="topol.top",
            log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    Modified function of gromacs.solvate(). cp is usually "box.gro"

    Args:
        cp (str): structure file ~ solute:  gro pdb tpr (g96 brk ent esp)
        cs (str):
          | structure file ~ solvent: gro pdb tpr (g96 brk ent esp)
          | Note: "spc216.gro" is used for all 3-point water models
        o (str):
          | gro pdb (g96 brk ent esp)
          | default case: save in same directory as cp
          | special case: if o is relative/absolute path -> save there
        p (str): topology file: topol.top
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/solvate.log>
          | False: save log file as <logs/solvate_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        maxsol (int):
          | maximum number of solvent molecules to add if they fit in the box.
          | 0 (default): ignore this setting
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.solvate.help()
        - terminal -> gmx solvate -h

    Returns:
        o_file (str)
            realpath of output file
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
        if not log_overwrite:
            log_names = [f"{logdir}/solvate_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def genion(s, o, p="topol.top", input="SOL", pname="NA", nname="CL", conc=0.15,
           neutral=True, log=True, log_overwrite=False, verbose=True, **kwargs):
    """
    Modified fuction of gromacs.genion().

    Gromacs info:
        gmx genion randomly replaces solvent molecules with monoatomic ions.
        The group of solvent molecules should be continuous and all molecules
        should have the same number of atoms.

    Args:
        s (str): structure file: tpr
        o (str): output file: gro pdb (g96 brk ent esp)
        p (str): topology file: topol.top
        input (str):
          | selection grp
          | default: 13 (SOL)
        pname (str): positive ion name
        nname (str): negative ion name
        conc (float): add salt concentration (mol/liter) and rescale to box size
        neutral (bool): add enough ions to neutralize the system. These ions are
          added on top of those specified with -np/-nn or -conc
        log (bool): save log file
        log_overwrite (bool):
          | True: save log file as <logs/genion.log>
          | False: save log file as <logs/genion_{i}.log> for i=1,...,999
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.genion.help()
        - terminal -> gmx genion -h

    Returns:
        o_file (str)
            realpath of output file
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
        if not log_overwrite:
            log_names = [f"{logdir}/genion_{i}.log" for i in range(1, 1000)]
            for name in log_names:
                if name in glob.glob(f"{logdir}/*.log"):
                    continue  # skip existing log_name
                else:
                    logfile = name
                    break  # stop at first new log_name
        with open(logfile, "w") as fout:
            _misc.cprint(f"Saved log as: {logfile}", cfg.cprint_color)
            fout.write(f"STDERR:\n\n{stderr}\n")
            fout.write(f"STDOUT:\n\n{stdout}\n")
            clean_up(logdir, verbose=False)
    return o_file


def mdrun(verbose=True, **kwargs):
    """
    Alias fuction of gromacs.mdrun().

    .. Note:: output can be printed only after mdrun has finished. To see
      realtime updates invoke the command using "!gmx_mpi mdrun <parameters>"
      instead

    Args:
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        deffnm (str): default filename
        nsteps (int): maximum number of steps (used instead of mdp setting)

    .. Hint:: Find more valid Keyword Args via

       - python -> gromacs.mdrun.help()
       - terminal -> gmx mdrun -h
    """
    # GromacsWrapper
    with _misc.HiddenPrints(verbose=verbose):
        stdin, stdout, stderr = gromacs.mdrun(v=True, **kwargs)
        print(stderr)
        print()
        print(stdout)
    return


def trjconv(s, f, o="default", odir="./", sel="protein", verbose=True, **kwargs):
    """
    Modified function of gromacs.trjconv().

    Gromacs info:
        gmx trjconv can convert trajectory files in many ways

        - from one format to another
        - select a subset of atoms
        - center atoms in the box
        - fit atoms to reference structure
        - reduce the number of frames
        - etc.

    Args:
        s (str): structure: tpr (gro g96 pdb brk ent)
        f (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str): trajectory:
          | xtc (trr gro g96 pdb tng)
          | "default": <f>.xtc -> <f>_<sel>.xtc
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        sel (str):
          | selection string (case independant)
          | "system" or "all": all atoms
          | "protein": protein atoms
          | "ca" or "calpha": CA atoms
          | "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        cprint_color (str)

    .. Hint:: Find more valid Keyword Args via

       - python -> gromacs.trjconv.help()
       - terminal -> gmx trjconv -h

    Returns:
        o_file (str)
            realpath of output file
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
        o (str, list):
          | filenames of new structure and new trajectory (both selection only)
          | "default" (str):
          |    <tpr>.tpr -> <xtc>_<sel>.tpr
          |    <xtc>.xtc -> <xtc>_<sel>.xtc
          | [os, of] (list):
          |    os (str): filename of new structure (selection only)
          |    of (str): filename of new trajectory (selection only)
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        tu (str): time unit: fs ps ns us ms s
        sel (str):
          | selection string(case independant)
          | "system" or "all": all atoms
          | "protein": protein atoms
          | "ca" or "calpha": CA atoms
          | "bb" or "backbone": backbone atoms
        pbc (str):
          | PBC treatment: none mol res atom nojump cluster whole
          | "mol": puts the center of mass of molecules in the box. requires structure file s.
          | "nojump": checks if atoms jump across the box and then puts them back
            (i.e. all molecules will remain whole)
          | (see gromacs help text for other descriptions)
        center (bool): center atoms in box
        verbose (bool): print/mute gromacs messages

    .. Hint:: Find more valid Keyword Args via

        - python -> gromacs.trjconv.help()
        - terminal -> gmx trjconv -h

    Returns:
        tpr_file (str)
            realpath of new tpr file (selection only)
        xtc_file (str)
            realpath of new xtc file (selection only)
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
    Modified function of gromacs.rms(). Calculate backbone RMSD.

    Args:
        ref (str): reference structure: pdb (tpr gro g96 brk ent)
        xtc (str): trajectory: xtc (trr cpt gro g96 pdb tng)
        o (str):
          | xvg
          | "default": rmsd.xvg
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        tu (str): time unit: fs ps ns us ms s
        sel (str):
          | selection string (case independant)
          | "system" or "all": all atoms
          | "protein": protein atoms
          | "ca" or "calpha": CA atoms
          | "bb" or "backbone": backbone atoms
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        cprint_color (str)

    .. Hint:: Find valid Keyword Args via

        - python -> gromacs.rms.help()
        - terminal -> gmx rms -h

    Returns:
        o_file (str)
            realpath of output file
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
                      ignh=True, verbose=True, **kwargs):
    """
    Creates ref structure of input structure via pdb2gmx to fix atom count by
    applying force field.

    Args:
        f (str): input structure file: pdb gro tpr (g96 brk ent esp)
        o (str): output structure file: pdb gro trp (g96 brk ent esp)
        odir (str):
          | output directory
          | special case: odir is ignored when o is relative/absolute path
        ff (str):
          | force field (see <gromacs_path>/top/<ff> for valid inputs)
          | "amber99sb-ildn" etc. for proteins
          | "amber14sb_OL15" etc. for RNAs
        water (str): water model
        ignh (bool): ignore hydrogen
        verbose (bool): print/mute gromacs messages

    Keyword Args:
        cprint_color (str)

    .. Hint:: Find valid Keyword Args via

        - python -> gromacs.pdb2gmx.help()
        - terminal -> gmx pdb2gmx -h

    Returns:
        o_file (str)
            realpath of output file
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
        gromacs.pdb2gmx(f=f, o=o, ff=ff, water=water.lower(), ignh=ignh, **kwargs)

    # save message
    o_file = _misc.realpath(o)
    _misc.cprint(f"Saved file as: {o_file}", cfg.cprint_color)
    return o_file

################################################################################
################################################################################
### COMPLEX BUILDER


def create_complex(f=[], o="complex.gro", verbose=True):
    """
    Create complex using multiple .gro files. First .gro file in <f> must have
    largest box.

    Args:
        f (list): list of gro files
        o (str): output structure file: gro
        verbose (bool)

    Returns:
        o_file (str)
            realpath of output file
    """
    if not isinstance(f, list):
        raise _misc.Error("Wrong data type: f must be list.")
    if len(f) < 2:
        raise _misc.Error("Not enough input files: f must contain atleast 2 .gro files.")
    for gro in f:
        if _misc.get_extension(gro) != ".gro":
            raise _misc.Error("Wrong extension: f does not contain .gro files.")

    for ndx, file in enumerate(f):
        with open(file, "r") as handle:
            read = handle.readlines()

        if ndx == 0:
            data = list(read)

        if ndx != 0:
            temp = f"{int(data[1]) + int(read[1])}\n"           # fix atom count
            if len(temp) < max(len(data[1]), len(read[1])):     # fix atom count position (gro has fixed format)
                temp = f" {temp}"
            data[1] = temp
            box = data.pop(-1)

            # test if 2nd box is smaller
            box_size_1 = [float(x) for x in box.split()[:3]]
            box_size_2 = [float(x) for x in read[-1].split()[:3]]
            box_size_1 = np.prod(box_size_1)  # get box volume via a*b*c
            box_size_2 = np.prod(box_size_2)
            if box_size_1 < box_size_2:
                raise _misc.Error(f"Wrong order of .gro files: box size of {f[0]} is smaller than {f[ndx]}. Type help({create_complex.__module__}.{create_complex.__name__})")

            # append data
            for line in read[2:-1]:
                data.append(line)

            #append box size of f[0]
            data.append(box)

    with open(o, "w") as handle:
        for line in data:
            handle.write(line)

    o_file = _misc.realpath(o)
    if verbose:
        _misc.cprint(f"Saved complex as: {o_file}", "blue")

    return o_file


def extend_complex_topology(ligand_name, ligand_itp, ligand_prm, ligand_nmol, top="topol.top", top_out="topol_complex.top", verbose=True):
    """
    Extend an existing .top file.

    Args:
        ligand_name (str): ligand name
        ligand_itp (str): ligand .itp file
        ligand_prm (str): ligand .prm file
        ligand_nmol (int): number of ligand molecules
        top (str): topology input file (will be extended)
        top_out (str): topology output file
        verbose (bool)

    Returns:
        o_file (str)
            realpath of output file
    """

    with open(top, "r") as handle:
        data = handle.readlines()

    with open(top_out, "w") as handle:
        # add very first ligand entry
        if "; Include ligand topology\n" not in data:
            for ndx, line in enumerate(data):
                # insert itp
                if "; Include water topology\n" in line:
                    handle.write(f'''; Include ligand topology\n#include "{ligand_itp}"\n''')
                    handle.write("\n")
                # insert prm
                if "forcefield.itp" in line:
                    handle.write(line)
                    handle.write(f'''\n; Include ligand parameters\n#include "{ligand_prm}"\n''')
                    continue
                handle.write(line)

        # add additional ligand entry
        else:
            itp_loc = False
            prm_loc = False
            for ndx, line in enumerate(data):
                # insert itp
                if "; Include ligand topology\n" in line:
                    itp_loc = True
                elif itp_loc and line == "\n":
                    handle.write(f'''#include "{ligand_itp}"\n''')
                    itp_loc = False
                # insert prm
                if "; Include ligand parameters\n" in line:
                    prm_loc = True
                if prm_loc and line == "\n":
                    handle.write(f'''#include "{ligand_prm}"\n''')
                    prm_loc = False
                handle.write(line)

        # insert last line (ligand name, ligand_nmol) with correct width
        len_last_line = len(data[-1])
        whitespace = " "
        temp = f"{ligand_name}{whitespace}{ligand_nmol}\n"
        while len_last_line != len(temp):
            whitespace += " "
            temp = f"{ligand_name}{whitespace}{ligand_nmol}\n"
        handle.write(temp)

    o_file = _misc.realpath(top_out)
    if verbose:
        _misc.cprint(f"Saved extended complex topology as: {o_file}", "blue")

    return o_file


def create_positions_dat(box_vec=[25, 25, 25],
                         nmol=range(100, 110, 10),
                         verbose=True):
    """
    Create folder with position.dat files. Each file contains <nmol> copies of
    the box center. Use in combination with "gmx insert-molecule module"


    Args:
        box_vec (list): box vectors
        nmol (int, tuple, list, range): number of molecules to add (replaces -nmol from
          insert-molecule cmd)
        verbose (bool)

    Returns:
        file_dir (str)
            realpath of folder containing positions.dat files

    Example:
        | >> gmx insert-molecules -f "box_25.gro" -ci "../model_ATP/atp_ini.pdb"
        | -o "complex.gro" -ip "positions_dat/positions_100.dat" -dr 5 5 5 -try 100
    """
    file_dir = _misc.mkdir("./positions_dat")
    if isinstance(nmol, int):
        nmol = [nmol]
    elif isinstance(nmol, (list, tuple, range)):
        pass
    else:
        raise _misc.Error("<nmol> must be int, tuple, list or range object.")
    for n in nmol:
        with open(f"{file_dir}/positions_{n}.dat", "w") as handle:
            for i in range(n):
                handle.write(f"{box_vec[0]/2}\t{box_vec[1]/2}\t{box_vec[2]/2}\t\n")
    if verbose:
        _misc.cprint(f"Dumped positions.dat files into folder: {file_dir}", "blue")
    return file_dir
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
