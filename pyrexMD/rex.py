# @Author: Arthur Voronin
# @Date:   05.05.2021
# @Filename: rex.py
# @Last modified by:   arthur
# @Last modified time: 16.09.2021


"""
.. hint:: This module contains functions related to (contact-guided) Replica
    Exchange Molecular Dynamics, mainly for automating and speeding up the
    simulation setup.

Example:
--------

.. code-block:: python

    import pyrexMD.misc as misc
    import pyrexMD.rex as rex
    import pyrexMD.topology as top

    decoy_dir = "path/to/decoy/directory"

    # create rex_i directories and assign decoys
    rex.assign_best_decoys(decoy_dir)
    rex_dirs = rex.get_REX_DIRS()

    # check for consistent topology
    rex.check_REX_PDBS(decoy_dir)

    # copy mdp files (min.mdp, nvt.mdp, npt.mdp, rex.mdp) into working directory
    misc.cp("path/to/mdp/files", ".")

    # get parameters for fixed box size and solvent molecules
    boxsize, maxsol = rex.WF_get_system_parameters(wdir="./rex_0_get_system_parameters/")

    # create systems for each replica
    rex.WF_REX_setup(rex_dirs=rex_dirs, boxsize=boxsize, maxsol=maxsol)

    # minimize
    rex.WF_REX_setup_energy_minimization(rex_dirs=rex_dirs, nsteps=10, verbose=False)

    # add bias contacts (RES pairs defined in DCA_fin)
    top.DCA_res2atom_mapping(ref_pdb=<ref_pdb>, DCA_fin=<file_path>, n_DCA=50, usecols=(0,1))
    top.DCA_modify_topology(top_fin="topol.top", DCA_used_fin=<file_path> , k=10, save_as="topol_mod.top")

    # prepare temperature distribution
    rex.prep_REX_temps(T_0=280, n_REX=len(rex_dirs), k=0.006)

    # create mdp and tpr files
    rex.prep_REX_mdp(main_dir="./", n_REX=len(rex_dirs))
    rex.prep_REX_tpr(main_dir="./", n_REX=len(rex_dirs))

    # next: upload REX MD run files on HPC and execute production run

Content:
--------
"""

from MDAnalysis import Universe
import os
import glob
import numpy as np
import pyrexMD.topology as _top
import pyrexMD.gmx as _gmx
import pyrexMD.misc as _misc
from tqdm.notebook import tqdm


def apply_ff_best_decoys(best_decoys_dir, n_decoys=None, odir="./PDBID_best_decoys_ref",
                         create_dir=True, verbose=False, **kwargs):
    """
    Apply forcefield on best decoys and save as <filename>_ref.pdb.

    Args:
        best_decoys_dir (str): directory with

          - best decoys (output of cluster.rank_cluster_decoys() -> cluster.copy_cluster.decoys())
          - decoy_scores.log (output of cluster.log_cluster.decoys())
        n_decoys (None, int):
          | None: use all decoys from best_decoys_dir
          | int: use this number of decoys from best_decoys_dir
        odir (str):
          | output directory
          | Note: if "PDBID" in <odir> and no <pdbid> kwarg is passed -> find and replace "PDBID" in <odir> automatically based on filenames
        create_dir (bool)
        verbose (bool)

    Keyword Args:
        pdbid (str)
        logfile (str): logfile name (default: "decoy_scores.log")
        water (str): water model (default: "tip3p")
        input (str): forcefield number (default: "6")
        cprint_color (str)

    Returns:
        odir (str)
            output directory with ref pdb files (forcefield is applied)
    """
    default = {"pdbid": "PDBID",
               "logfile": "decoy_scores.log",
               "water": "tip3p",
               "input": "6",
               "cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)

    # replace PDBID if passed
    if "PDBID" in odir and cfg.pdbid != "PDBID":
        odir = odir.replace("PDBID", cfg.pdbid)
    # find and replace PDBID if not passed
    if "PDBID" in odir and cfg.pdbid == "PDBID":
        min_length = 999999
        for item in glob.glob(f"{best_decoys_dir}/*pdb"):
            filename = _misc.get_base(item)
            if len(filename) < min_length:
                min_filename = filename
                min_length = len(filename)
        cfg.pdbid = _misc.get_PDBid(min_filename).upper()
        odir = odir.replace("PDBID", cfg.pdbid)

    if create_dir:
        odir = _misc.mkdir(odir)

    # apply ff
    DECOY_PATHS = [f"{best_decoys_dir}/{item}" for item in
                   _misc.read_file(f"{best_decoys_dir}/{cfg.logfile}",
                                   usecols=0, skiprows=1, dtype=str)]
    if n_decoys is not None:
        DECOY_PATHS = DECOY_PATHS[:n_decoys]
    for item in tqdm(DECOY_PATHS):
        with _misc.HiddenPrints(verbose=verbose):
            _gmx.get_ref_structure(item, odir=odir, water=cfg.water,
                                   input=cfg.input, verbose=verbose)
    _gmx.clean_up("./", verbose=False)
    _gmx.clean_up(odir, verbose=False)

    # copy and fix log
    with open(f"{best_decoys_dir}/{cfg.logfile}", "r") as old_log, open(f"{odir}/{cfg.logfile}", "w") as new_log:
        count = 0
        for line in old_log:
            if ".pdb" in line:
                count += 1
            new_log.write(line.replace(".pdb", "_ref.pdb"))

            if n_decoys is not None and n_decoys == count:
                break

    if verbose:
        _misc.cprint(f"Saved files to: {odir}", cfg.cprint_color)
    return odir


def assign_best_decoys(best_decoys_dir, rex_dir="./", verbose=False, **kwargs):
    """
    Assigns decoys based on ranking taken from <best_decoys_dir>/decoy_scores.log
    to each rex subdirectory rex_1, rex_2, ...
    Also creates a 'rex_0_get_system_parameters' directory, which is a copy of rex_1.

    Args:
        best_decoys_dir (str): directory with

          - best decoys (output of cluster.rank_cluster_decoys() -> cluster.copy_cluster.decoys())
          - decoys_score.log (output of cluster.log_cluster.decoys())
        rex_dir (str): rex directory with folders rex_1, rex_2, ...
        verbose (bool)

    Keyword Args:
        cprint_color (str)
        realpath (bool): prints realpaths
    """
    default = {"cprint_color": "blue",
               "realpath": False}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    DECOY_PATHS = [f"{best_decoys_dir}/{item}" for item in
                   _misc.read_file(f"{best_decoys_dir}/decoy_scores.log",
                                   usecols=0, skiprows=1, dtype=str)]

    for ndx, item in enumerate(DECOY_PATHS, start=1):
        target = _misc.joinpath(rex_dir, f"rex_{ndx}/")
        _misc.mkdir(target)
        _misc.cp(item, target, verbose=verbose)
        if ndx == 1:    # rex_0_get_system_parameters: copy of rex_1
            target = _misc.joinpath(rex_dir, "rex_0_get_system_parameters/")
            _misc.mkdir(target)
            _misc.cp(item, target, verbose=verbose)

    if not verbose:
        if cfg.realpath:
            _misc.cprint(f"Copied source files to directories: {os.path.realpath(rex_dir)}/rex_<i> (i=1,...,{len(DECOY_PATHS)})", cfg.cprint_color)
        else:
            _misc.cprint(f"Copied source files to directories: {os.path.relpath(rex_dir)}/rex_<i> (i=1,...,{len(DECOY_PATHS)})", cfg.cprint_color)
    return


def get_REX_DIRS(main_dir="./", realpath=True):
    """
    get list with REX DIRS.

    Args:
        main_dir (str): directory with folders rex_1, rex_2, etc.
        realpath (bool): return realpath

    Returns:
        REX_DIRS (list)
            list with REX directory paths
    """
    n_REX = 300   # hardcoded but will be filtered automatically
    REX_DIRS = _misc.flatten_array([glob.glob(os.path.realpath(main_dir) + f"/rex_{i}") for i in range(0, n_REX+1)
                                    if len(glob.glob(os.path.realpath(main_dir) + f"/rex_{i}/*.pdb")) != 0])   # remove emptry entries
    if not realpath:
        REX_DIRS = [os.path.relpath(item) for item in REX_DIRS]

    # filter for directories
    REX_DIRS = [item for item in REX_DIRS if os.path.isdir(item)]
    return REX_DIRS


def get_REX_PDBS(main_dir="./", realpath=True):
    """
    get list with REX PDBS.

    Args:
        main_dir (str): directory with folders rex_1, rex_2, etc.
        realpath (bool): return realpath

    Returns:
        REX_PDBS (list)
            list with PDB paths within folders rex_1, rex_2, etc.
    """
    n_REX = 300   # hardcoded but will be filtered automatically
    REX_PDBS = _misc.flatten_array([glob.glob(os.path.realpath(main_dir) + f"/rex_{i}/*.pdb") for i in range(0, n_REX+1)
                                    if len(glob.glob(os.path.realpath(main_dir) + f"/rex_{i}/*.pdb")) != 0])   # remove emptry entries
    if not realpath:
        REX_PDBS = [os.path.relpath(item) for item in REX_PDBS]

    # filter for files
    REX_PDBS = [item for item in REX_PDBS if os.path.isfile(item)]
    return REX_PDBS


def check_REX_PDBS(REX_PDBS, ref_pdb=None, sel="protein", verbose=True, **kwargs):
    """
    Tests if all REX PDBS have equal topologies (RES, ID, NAME arrays). Uses ref_pdb
    as template for tests if passed, otherwise uses first REX pdb as template.

    .. Note ::

        - if target is known -> apply ff -> save as ref -> use as ref
        - if target is unknown -> use ref_pdb=None or use one of REX_PDBS -> apply ff -> use as ref
        - use check2_REX_PDBS() for debugging if check_REX_PDBS() fails.

    Args:
        REX_PDBS (list): output of get_REX_PDBS()
        ref_pdb (None, str):
          | reference pdb
          | None: use ref_pdb as template for tests
          | str: use first REX pdb as template for tests
        sel (str): selection string
        ref_is_known(bool):
          | True:  use ref_pdb as template for tests
          | False: use first REX DPB as template for tests
        verbose (bool)
    """
    if ref_pdb is None:
        # treat first pdb as "template pdb" with target arrays
        template_pdb = REX_PDBS[0]
        if verbose:
            _misc.cprint("Using first REX PDB as template for tests...", "blue")
    else:
        # treat ref pdb as "template pdb" with target arrays
        template_pdb = ref_pdb
        if verbose:
            _misc.cprint("Using ref_pdb as template for tests...", "blue")

    template_RESID, template_RESNAME, template_ID, template_NAME = _top.parsePDB(template_pdb, sel=sel)

    for pdb_file in tqdm(REX_PDBS):
        RESID, RESNAME, ID, NAME = _top.parsePDB(template_pdb, sel=sel)

        if template_RESID != RESID or template_ID != ID or template_NAME != NAME:
            raise _misc.Error(f"Parsed arrays of {template_pdb} do not match with {pdb_file}. Use check2_REX_PDBS() for debugging.")

    _misc.cprint("Check passed: All REX PDBs have equal RES, ID, NAME arrays", "green")
    return


def check2_REX_PDBS(REX_PDBS, ref_pdb=None, verbose=False):
    """
    Used for debbuging if check_REX_PDBS() fails.

    Tests if all REX PDBS have equal topologies (RES, ID, NAME arrays). Uses ref_pdb
    as template for tests if passed, otherwise uses first REX pdb as template.

    .. Note ::

        - if target is known -> apply ff -> save as ref -> use as ref
        - if target is unknown -> use ref_pdb=None or use one of REX_PDBS -> apply ff -> use as ref

    Args:
        REX_PDBS (list): output of get_REX_PDBS()
        ref_pdb (None, str):
          | reference pdb
          | None: use ref_pdb as template for tests
          | str: use first REX pdb as template for tests
        sel (str): selection string
        ref_is_known(bool):
          | True:  use ref_pdb as template for tests
          | False: use first REX DPB as template for tests
        verbose (bool)
    """
    if ref_pdb is None:
        ref_pdb = REX_PDBS[0]
    _misc.cprint(f"ref pdb: {ref_pdb}", "blue")

    check_passed = True
    for test_pdb in tqdm(REX_PDBS):
        x1 = Universe(ref_pdb)
        x2 = Universe(test_pdb)

        if np.all(x1.residues.resnames == x2.residues.resnames) == False:
            check_passed = False
            print("###########################################################################################")
            _misc.cprint(f"RES NAMES are not equal ({test_pdb}):", "red")
            print(x1.residues.resnames == x2.residues.resnames)
        elif verbose:
            print("###########################################################################################")
            _misc.cprint(f"RES NAMES are equal ({test_pdb}).", "green")
        if np.all(x1.atoms.names == x2.atoms.names) == False:
            check_passed = False
            _misc.cprint(f"ATOM NAMES are not equal ({test_pdb}):", "red")
            for i in range(len(x1.atoms.names)):
                if x1.atoms.names[i] != x2.atoms.names[i]:
                    _misc.cprint(f"resid1: {x1.atoms.resids[i]}    resid2: {x2.atoms.resids[i]} || atom-id1: {x1.atoms.ids[i]}    atom-id2: {x2.atoms.ids[i]} || atom-name1:{x1.atoms.names[i]}    atom-name2:{x2.atoms.names[i]}")
        elif verbose:
            _misc.cprint(f"ATOM NAMES are equal ({test_pdb}).", "green")

    if check_passed:
        _misc.cprint("Check passed: All REX PDBs have equal RES, ID, NAME arrays", "green")
    else:
        _misc.cprint("Check failed.", "red")
    return


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
        verbose (bool)prep_REX_tpr

    Returns:
        boxsize (float)
            suggested boxsize parameter
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
            _misc.cprint(f"suggested box size: {boxsize}", "green")
    return boxsize


def WF_getParameter_maxsol(logfile="./logs/solvate.log", maxsol_reduce=50, verbose=True):
    """
    Read <logfile> and suggest a 'max solution' parameter for REX simulations.

    Args:
        logfile (str): path to <editconf.log> containing the line 'system size: X Y Z'
        maxsol_reduce (int):
          | reduce max solution number taken from <logfile>
          | -> guarantees fixed solution number for different start configurations
        verbose (bool)

    Returns:
        maxsol (int)
            suggested max solution parameter
    """
    maxsol = None
    with open(logfile, "r") as fin:
        for line in fin:
            if "Number of solvent molecules" in line:
                s = line.split()
                maxsol = [int(x)-maxsol_reduce for x in s if x.isdigit()][0]

                if verbose:
                    _misc.cprint(f"Reading logfile: {logfile}")
                    _misc.cprint(line, "blue", end="")
                    _misc.cprint(f"suggested max solution: {maxsol}", "green")

    if maxsol is None:
        raise _misc.Error("No solution parameter found.")
    return maxsol


def WF_get_system_parameters(wdir, verbose=False, **kwargs):
    """
    Get system parameters "boxsize" and "maxsol" for system in working directory
    wdir. Typically, wdir is supposed to be the folder "rex_0_get_system_parameters"
    which is created during the Workflow for setup of REX MD simulations during
    the execution of rex.assign_best_decoys().

    Args:
        wdir (str): working directory
        verbose (bool)

    Keyword Args:
        d (int): distance between model and system edges. Defaults to 2.
        bt (str): box type. Defaults to "cubic".

    Returns:
        boxsize (float)
            suggested value for maximum boxsize
        maxsol (int)
            suggested value for maximum solvent molecules
    """
    default = {"d": 2,
               "bt": "cubic"}
    cfg = _misc.CONFIG(default, **kwargs)
    ########################################################################
    if not os.path.isdir(wdir):
        raise ValueError("The working directory wdir does not exist or is not a directory.")

    # save cwd and go to wdir
    cwd = os.getcwd()
    wdir = _misc.cd(wdir)

    pdbs = glob.glob("*.pdb")
    ref_pdbs = glob.glob("*_ref.pdb")
    if len(ref_pdbs) > 1:
        raise ValueError(f"Found multiple '_ref.pdbs' in working directory {wdir}.")
    if len(ref_pdbs) == 1:
        decoy_pdb = ref_pdbs[0]
    if len(ref_pdbs) == 0:
        _misc.cprint("Found no '_ref.pdbs' in working directory {wdir}.")
    if len(pdbs) > 1:
        raise ValueError(f"Found multiple pdbs in working directory {wdir}.")
    if len(pdbs) == 1:
        decoy_pdb = pdbs[0]
    if len(pdbs) == 0:
        raise ValueError("Found no pdbs in working directory {wdir}.")

    with _misc.HiddenPrints(verbose=verbose):
        # generate topology
        protein_gro = _gmx.pdb2gmx(f=decoy_pdb, verbose=False)

        # generate box
        box_gro = _gmx.editconf(f=protein_gro, o="box.gro", bt=cfg.bt, d=cfg.d, verbose=False)

    # get boxsize
    boxsize = WF_getParameter_boxsize("./logs/editconf_1.log")

    with _misc.HiddenPrints(verbose=verbose):
        # apply fixed boxsize
        box_gro = _gmx.editconf(f=protein_gro, o="box.gro", bt=cfg.bt, box=boxsize, c=True, verbose=False)

    if boxsize == WF_getParameter_boxsize("./logs/editconf_2.log"):
        _misc.cprint(f"Success: new box vectors have fixed size: {boxsize}", "green")
    else:
        _misc.cprint(f"Check above if new box vectors have fixed size: {boxsize}", "red")

    with _misc.HiddenPrints(verbose=verbose):
        # generate solvent
        _ = _gmx.solvate(cp=box_gro, verbose=verbose)

    # get maximum solvent number
    maxsol = WF_getParameter_maxsol("./logs/solvate_1.log")

    # go back to cwd
    _misc.cd(cwd)
    return boxsize, maxsol


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
        _misc.cprint("\nGenerating topology...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            protein_gro = _gmx.pdb2gmx(f=decoy_pdb, verbose=False)

        # 2) generate box
        _misc.cprint(f"Generating box with fixed size ({boxsize}) ...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            _gmx.editconf(f=protein_gro, o="box.gro", bt="cubic", box=boxsize, c=True, verbose=verbose_gmx)

        # 3) generate solvent
        _misc.cprint(f"Generating solvent with fixed solvent molecules ({maxsol})...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            _gmx.solvate(cp="box.gro", maxsol=maxsol, verbose=verbose_gmx)

        # 4) generate ions
        _misc.cprint("Generating ions...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            _gmx.grompp(f="../ions.mdp", o="ions.tpr", c="solvent.gro", p="topol.top", verbose=verbose_gmx)
            _gmx.genion(s="ions.tpr", o="ions.gro", p="topol.top", verbose=False)

    _misc.cprint("#######################################################################################")
    _misc.cd("..")
    _misc.cprint("Finished setup of all REX DIRS (skipped energy minimization).", "green")
    return


def WF_REX_setup_energy_minimization(rex_dirs, nsteps=None, verbose=False):
    """
    Workflow: REX energy minimization

    Args:
        rex_dirs (list): list with rex_dirs (output of rex.get_REX_DIRS())
        nsteps (None,int):
          | maximum number of steps
          | None: use .mdp option
          | int: use instead of .mdp option
        verbose (bool): show/hide gromacs output
    """
    for rex_dir in rex_dirs:
        _misc.cprint("#######################################################################################")
        _misc.cd(rex_dir)

        # 5) energy minimization
        _misc.cprint("Performing energy minimization...", "red")
        with _misc.HiddenPrints(verbose=verbose):
            _ = _gmx.grompp(f="../min.mdp", o="em.tpr", c="ions.gro", p="topol.top")
            if nsteps is None:
                _gmx.mdrun(deffnm="em", verbose=verbose)
            else:
                _gmx.mdrun(deffnm="em", nsteps=nsteps, verbose=verbose)
            _gmx.clean_up("./", verbose=False)

    _misc.cprint("#######################################################################################")
    _misc.cd("..")
    _misc.cprint("Finished energy minimization of all REX DIRS.", "green")
    return


def create_special_group_ndx(ref_pdb, sel="name CA", save_as="special_group.ndx"):
    """
    Create index file for special group.

    Args:
        ref_pdb (str): PDB file (after applying force field)
        sel (str): selection string
        save_as (str)
    """
    RESID, RESNAME, ATOM, NAME = _top.parsePDB(ref_pdb, sel=sel)
    with open(save_as, "w") as fout:
        fout.write(f"[ {sel} ]\n")
        for i in range(len(ATOM)):
            fout.write(f"\t{ATOM[i]}")
        fout.write("\n")
        realpath = os.path.realpath(save_as)
        _misc.cprint(f"Saved file as: {realpath}")
    return


def create_pull_group_ndx(ref_pdb, sel="name CA", save_as="pull_group.ndx"):
    """
    Create index file for pull group.

    Args:
        ref_pdb (str): PDB file (after applying force field)
        sel (str): selection string
        save_as (str)
    """
    RESID, RESNAME, ATOM, NAME = _top.parsePDB(ref_pdb, sel=sel)
    with open(save_as, "w") as fout:
        for i in range(len(RESID)):
            fout.write(f"[ pull_res{RESID[i]} ]\n")
            fout.write(f"\t{ATOM[i]}\n")
        realpath = os.path.realpath(save_as)
        _misc.cprint(f"Saved file as: {realpath}")
    return


################################################################################
################################################################################
### prep REX run files functions


def prep_REX_temps(T_0=None, n_REX=None, k=None):
    """
    Prepare REX temperatures.

    Args:
        T_0 (None/float): starting temperature (Kelvin)
        n_REX (None/int): number of replica
        k (None/float): temperature distrubtion scaling factor
    """
    T1 = []		# list with temperatures for method 1
    T2 = []		# list with temperatures for method 2

    if T_0 is None:
        T_0 = int(input("Enter starting temperature T_0 (Kelvin): "))
    else:
        print("Enter starting temperature T_0 (Kelvin): ", T_0)
    if n_REX is None:
        n_REX = int(input("Enter replica number n_REX: "))
    else:
        print("Enter replica number n_REX: ", n_REX)
    if k is None:
        k = float(input("Enter scaling factor k: "))
    else:
        print("Enter scaling factor k: ", k)

    # method 1: same function for all replica
    print("\nMethod 1:")
    print("REX Temperature Distribution: T_i = T_0*exp(k*i)")

    with open("rex_temps.log", "w") as fout:
        _misc.cprint(f"Saved log as: {os.path.realpath('rex_temps.log')}", "blue")
        fout.write("Function: T_i = {}*exp({}*i)\n".format(T_0, k))
        fout.write("Temperatures:\n")
        for i in range(0, n_REX):
            T_i = format(1.00*T_0*np.exp(k*i), ".2f")
            T1.append(T_i)
            fout.write("{}, ".format(T_i))

    # method 2: same as method 1 but with adjusting the coefficients ai
    # (ai increase spacing of higher temperatures to get lower exchange probabilities)

    a = []  # lin. stretching factors
    delta_a = 0.04

    # increase a by delta_a every 10 replica
    if n_REX % 10 == 0:
        for i in range(n_REX//10):
            if i == 0:
                a.append(1.00)
            else:
                a.append(round(a[-1]+delta_a, 2))
    else:
        for i in range((n_REX//10)+1):
            if i == 0:
                a.append(1.00)
            else:
                a.append(round(a[-1]+delta_a, 2))

    print("\nMethod 2")
    print("REX Temperature Distribution:")
    print("T_0 = {} K ; DELTA = T_0 * (exp(k*i)-exp(k*(i-1)))".format(T_0))
    print("T_i = T_(i-1) + a_i * DELTA")

    with open("rex_temps.log", "w") as fout:    # same fout name as method 1 -> will overwrite
        _misc.cprint(f"Saved log as: {os.path.realpath('rex_temps.log')}", "blue")
        fout.write("REX Temperature Distribution:\n")
        fout.write("T_0 = {} K ; DELTA = T_0 * (exp(k*i)-exp(k*(i-1)))\n".format(T_0))
        fout.write("T_i = T_(i-1) + a_j * DELTA\n")
        fout.write("\nChosen Parameter:\n")
        fout.write("k = {}\n".format(k))
        for i in range(len(a)):
            fout.write("a_{} = {:.2f} for i = {}..{} \n".format(i, a[i], i*10, i*10+9))

        fout.write("\nTemperatures:\n")
        for i in range(0, n_REX):
            if i == 0:
                T2.append(T_0)  # start value 300 K
            else:
                index = i//10  # find out which a[index] to use
                delta = T_0*(np.exp(k*i) - np.exp(k*(i-1)))
                T2.append(T2[i-1] + a[index]*delta)

        T2 = [format(x, ".2f") for x in T2]
        for i in range(len(T2)):
            fout.write("{}, ".format(T2[i]))

    # comparison of method 1 and 2
    print("\nTemperatures Method 1:")
    print(T1)
    print("\nTemperatures Method 2:")
    print(T2)

    # DELTA TEST
    DELTA1 = []
    DELTA2 = []
    DELTA_DELTA = []

    for i in range(len(T1)-1):
        DELTA1.append(format(float(T1[i+1])-float(T1[i]), ".2f"))
    print("\nDelta Temps Method 1 (DTM1):")
    print(DELTA1)

    for i in range(len(T2)-1):
        DELTA2.append(format(float(T2[i+1])-float(T2[i]), ".2f"))
    print("\nDelta Temps Method 2 (DTM2):")
    print(DELTA2)

    for i in range(len(DELTA1)):
        DELTA_DELTA.append(format(float(DELTA2[i])-float(DELTA1[i]), ".2f"))
    print("\nDelta_Delta (DTM2-DTM1):")
    print(DELTA_DELTA)


def prep_REX_mdp(main_dir="./", template="rex.mdp", n_REX=None, verbose=True):
    """
    Prepare REX mdp -> copy template and change tempereratures according to rex_temps.log

    Args:
        main_dir (str): main directory with rex_1, rex_2, etc.
        template (str): template file
        n_REX (None/int): number of replica
        verbose (bool)
    """
    with open("rex_temps.log", "r") as fin:
        for line in fin:
            # search for REX temperatures (should be last entry)
            s = line.split(",")
            REX_TEMPS = [x.lstrip() for x in s if "." in x and x.lstrip().replace(".", "", 1).isdigit()]

    # create N rex.mdp files and edit the ref_t line
    if n_REX is None:
        n_REX = int(input("Enter n_REX:"))
    else:
        print("Enter n_REX:", n_REX)
    print(f"Using {template} as template and changing temperatures according to rex_temps.log...")

    for i in range(1, n_REX+1):
        file_path = _misc.mkdir("rex_" + str(i))
        with open(_misc.joinpath(main_dir, template), "r") as fin, open(file_path + f"/{template}", "w") as fout:
            if verbose:
                print("Saved mdp file as: " + file_path + f"/{template}")

            for line in fin:
                if "ref_t" in line:
                    fout.write("ref_t   = %s       %s         ; reference temperature, one for each group, in K\n" % (REX_TEMPS[i-1], REX_TEMPS[i-1]))
                elif "gen_temp" in line:
                    fout.write("gen_temp = %s  ; temperature for Maxwell distribution\n" % (REX_TEMPS[i-1]))
                else:
                    fout.write(line)

    return


def prep_REX_tpr(main_dir="./", n_REX=None, verbose=False, **kwargs):
    """
    Prepare REX tpr.

    Args:
        main_dir (str): main directory with rex_1, rex_2, etc.
        n_REX (None/int): number of replica
        verbose (bool)

    Keyword Args:
        f (str)
        o (str)
        c (str)
        p (str)

    .. Note:: Keyword Args are parameter of gmx.grompp(f,o,c,r,p)
    """
    default = {"f": "rex.mdp",
               "o": "rex.tpr",
               "c": "em.gro",
               "p": "topol_mod.top"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    rex_dirs = get_REX_DIRS(main_dir)

    if n_REX is None:
        n_REX = int(input("Enter n_REX:"))
    else:
        print("Enter n_REX:", n_REX)
    _misc.cprint(f"preparing rex.tpr for {n_REX} replica.", "green")

    for ndx, rex_dir in enumerate(rex_dirs[:n_REX], start=1):
        _misc.cprint("#######################################################################################")
        _gmx.grompp(f=f"{rex_dir}/{cfg.f}", o=f"{rex_dir}/{cfg.o}",
                    c=f"{rex_dir}/{cfg.c}", p=f"{rex_dir}/{cfg.p}",
                    verbose=verbose)

    _misc.cprint(f"Finished REX tpr creation for REX DIRS (i=1..{n_REX}).", "green")
    return
