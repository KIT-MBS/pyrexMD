from __future__ import division, print_function
from builtins import next
from tqdm.notebook import tqdm
import myPKG.misc as _misc
import myPKG.gmx as _gmx
import numpy as np
import glob


def apply_ff_best_decoys(best_decoys_dir, odir="./PDBID_best_decoys_ref",
                         create_dir=True, verbose=False, **kwargs):
    """
    Apply forcefield on best decoys and save as <filename>_ref.pdb.

    Args:
        best_decoys_dir (str): directory with
            - best decoys (output of cluster.rank_cluster_decoys() -> cluster.copy_cluster.decoys())
            - decoy_scores.log (output of cluster.log_cluster.decoys())
        odir (str): output directory
            Note: if "PDBID" in <odir> and no <pdbid> kwarg is passed
                  -> find and replace "PDBID" in <odir> automatically based on filenames
        create_dir (bool)
        verbose (bool)

    Kwargs:
        pdbid (str)
        logfile (str): logfile name (default: "decoy_scores.log")
        water (str): water model (default: "tip3p")
        input (str): forcefield number (default: "6")
        cprint_color (str)

    Returns:
        odir (str): output directory with ref pdb files (forcefield is applied)
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
    for item in tqdm(DECOY_PATHS):
        with _misc.HiddenPrints(verbose=verbose):
            _gmx.get_ref_structure(item, odir=odir, water=cfg.water,
                                   input=cfg.input, verbose=verbose)
    _gmx.clean_up("./", verbose=False)
    _gmx.clean_up(odir, verbose=False)

    # copy and fix log
    with open(f"{best_decoys_dir}/{cfg.logfile}", "r") as old_log, open(f"{odir}/{cfg.logfile}", "w") as new_log:
        for line in old_log:
            new_log.write(line.replace(".pdb", "_ref.pdb"))

    if verbose:
        _misc.cprint(f"Saved files to: {odir}", cfg.cprint_color)
    return odir


def assign_best_decoys(best_decoys_dir, rex_dir="./", create_dir=True, verbose=False, **kwargs):
    """
    Assigns decoys based on ranking taken from <best_decoys_dir>/decoy_scores.log
    to each rex subdirectory rex_1, rex_2, ...

    Args:
        best_decoys_dir (str): directory with
            - best decoys (output of cluster.rank_cluster_decoys() -> cluster.copy_cluster.decoys())
            - decoys_score.log (output of cluster.log_cluster.decoys())
        rex_dir (str): rex directory with folders rex_1, rex_2, ...
        create_dir (bool)
        verbose (bool)

    Kwargs:
        cprint_color (str)
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)

    DECOY_PATHS = [f"{best_decoys_dir}/{item}" for item in
                   _misc.read_file(f"{best_decoys_dir}/decoy_scores.log",
                                   usecols=0, skiprows=1, dtype=str)]
    for ndx, item in enumerate(DECOY_PATHS, start=1):
        target = _misc.joinpath(rex_dir, f"rex_{ndx}")
        if create_dir:
            _misc.mkdir(target, verbose=verbose)
        _misc.cp(item, target, verbose=verbose)

    if not verbose:
        _misc.cprint(f"Copied source files to directories: {_misc.realpath(rex_dir)}/rex_<i> (i=1,...,{len(DECOY_PATHS)})", cfg.cprint_color)
    return


def get_REX_DIRS(main_dir="./", realpath=True):
    """
    get list with REX DIRS.

    Args:
        main_dir (str): directory with folders rex_1, rex_2, etc.
        realpath (bool): return realpath

    Returns:
        REX_DIRS (list): list with REX directory paths
    """
    n_REX = 300   # hardcoded but will be filtered automatically
    REX_DIRS = _misc.flatten_array([glob.glob(_misc.joinpath(main_dir, f"rex_{i}", realpath=realpath))
                                    for i in range(0, n_REX + 1)
                                    if len(glob.glob(_misc.joinpath(main_dir, f"rex_{i}", realpath=realpath))) != 0])

    REX_DIRS = [item for item in REX_DIRS if _misc.isdir(item)]
    return REX_DIRS


def get_REX_PDBS(main_dir="./", realpath=True):
    """
    get list with REX PDBS.

    Args:
        main_dir (str): directory with folders rex_1, rex_2, etc.
        realpath (bool): return realpath

    Returns:
        REX_PDBS (list): list with PDB paths within folders rex_1, rex_2, etc.
    """
    n_REX = 300   # hardcoded but will be filtered automatically
    REX_PDBS = _misc.flatten_array([glob.glob(_misc.joinpath(main_dir, f"rex_{i}/*.pdb", realpath=realpath))
                                    for i in range(0, n_REX + 1)
                                    if len(glob.glob(_misc.joinpath(main_dir, f"rex_{i}", realpath=realpath))) != 0])

    REX_PDBS = [item for item in REX_PDBS if _misc.isfile(item)]
    return REX_PDBS


def test_REX_PDBS(REX_PDBS, ref_pdb, ignh=True, verbose=True, **kwargs):
    """
    Test if all REX PDBS have equal RES, ATOM, NAME arrays.
    Uses ref as "template PDB".

    Args:
        REX_PDBS (list): output of get_REX_PDBS()
        ref_pdb (str): reference pdb
            if target is known: apply ff -> save as ref -> use as ref
            if target is unknown: use one of REX_PDBS -> apply ff -> use as ref
        ignh (bool): ignore hydrogen
        verbose (bool)

    Kwargs:
        cprint_color (None/str)
    """
    default = {"cprint_color": "green"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # treat first pdb as "template pdb" with target array lengths
    template_pdb = ref_pdb
    template_RES, template_ATOM, template_NAME = parsePDB_RES_ATOM_NAME(template_pdb, ignh=ignh)

    for pdb_file in REX_PDBS:
        RES, ATOM, NAME = parsePDB_RES_ATOM_NAME(pdb_file, ignh=ignh)

        if template_RES != RES or template_ATOM != ATOM or template_NAME != NAME:
            raise _misc.Error(f"Parsed arrays of {template_pdb} do not match with {pdb_file}.")
    if verbose:
        _misc.cprint("All tested PDBs have equal RES, ATOM, NAME arrays.", cfg.cprint_color)
    return


def parsePDB_RES_ATOM_NAME(fin, skiprows="auto", ignh=True):
    """
    Reads PDB file and returns the columns for residue, atom-number and atom-name as lists.
    Removes column entries which consist only of whitespace or newlines.

    Args:
        fin (str): PDB file
        skiprows (int): ignore header rows of fin
            -1 or "auto": auto detect
        ignh (bool): ignore hydrogen

    Returns:
        RES (list): residue column of PDB
        ATOM (list): atom column of PDB
        NAME (list): name column of PDB
    """
    # help function to ignh
    def HELP_atom_is_hydrogen(line):
        """
        help function: atom is hydrogen

        If first char of last line entry is "H":     return True  -> atom is hydrogen
        If first char of last line entry is not "H": return False -> atom is heavy atom (C,N,O,S,...)
        """
        if line[:4] == "ATOM":
            try:
                first_char = line[-5:].split()[0][0]
                if first_char == "H":
                    return True
                else:
                    return False
            except IndexError:  # some pdbs only write letters of heavy atoms in last column -> H atoms have blank columns
                return True

        #ignore lines without "ATOM" at start
        else:
            return False

    with open(fin, "r") as f:

        RES = []
        ATOM = []
        NAME = []

        # skip header rows
        if skiprows == -1 or skiprows == "auto":
            skiprows = _misc.autodetect_header(fin)
        for i in range(skiprows):
            next(f)
        for line in f:
            if line[:4] == "ATOM":  # parse only ATOM entries
                line_RES = line[22:26]
                line_ATOM = line[6:11]
                line_NAME = line[12:16]
                # remove all whitespace (empty colums ~ PDB format)
                line_RES = line_RES.replace(" ", "")
                line_ATOM = line_ATOM.replace(" ", "")
                line_NAME = line_NAME.replace(" ", "")
                # append to list
                if ignh == False:
                    # append all
                    RES.append(line_RES)
                    ATOM.append(line_ATOM)
                    NAME.append(line_NAME)
                elif not HELP_atom_is_hydrogen(line):
                    # append only heavy atoms
                    RES.append(line_RES)
                    ATOM.append(line_ATOM)
                    NAME.append(line_NAME)

        # double check: remove empty entries (might occur at beginning/ending of structures)
        RES = [int(item) for item in RES if item.isalnum()]
        ATOM = [int(item) for item in ATOM if item.isalnum()]
        NAME = [item for item in NAME if item.isalnum()]

    return RES, ATOM, NAME


def DCAREX_res2atom_mapping(ref_pdb, DCA_fin, n_DCA, usecols, n_bonds=1,
                            ref_skiprows="auto", DCA_skiprows="auto",
                            filter_DCA=True, save_log=True, **kwargs):
    """
    Get DCA contact mapping:
    Return lists of matching RES pairs and ATOM pairs for "DCA REX Workflow"

    Algorithm:
        - Read files line by line, skip header if skiprows is set correctly
            -- ref_pdb: get RES, ATOM, NAME
            -- DCA_fin: get DCA_pairs
        - for IJ in DCA_pairs:
            -- find matching CA atoms (CA_ij)
            -- append IJ to RES_pair, CA_ij to ATOM_pair
        - return RES_pair and ATOM_pair

    Args:
        ref_pdb (str): reference PDB (path)
        ref_skiprows (int): ignore header rows of ref
            -1 or "auto": auto detect
        DCA_fin (str): DCA file (path)
        n_DCA (int): number of DCA contacts which should be used
        usecols (tuple/list): columns containing the RES pairs in DCA_fin
        n_bonds (int/str): number of restraints per DCA contact
            "all", 0 or -1: bonds between all heavy atoms of RES pair ij
                    -> all permutations of heavy atoms per DCA pair
            1: bonds between  CA atoms of RES pair ij
                    -> 1 bond per DCA pair
            2: bonds between CA, CB atoms of RES pair ij
                    -> 2 bonds per DCA pair
            3: bonds between CA, CB, CG atoms of RES pair ij
                    -> 3 bonds per DCA pair
        DCA_skiprows (int): ignore header rows of DCA_fin
            -1 or "auto": auto detect
        filter_DCA (bool):
            True: ignore DCA pairs with |i-j| < 3
            False: use all DCA pairs w/o applying filter
        save_log (bool)

    Kwargs:
        cprint_color (None/str): colored print color
        pdbid (str):
            "auto" (default): detect PDBID based on ref_PDB path
        default_dir (str): "./"
        save_as (str): "PDBID_DCA_used.txt"
            detect and replace PDBID in kwarg <save_as> based on ref_PDB path
            if kwarg <pdbid> is "auto" (default).

    Returns:
        RES_PAIR (list)
        ATOM_PAIR (list)
    """
    default = {"cprint_color": "blue",
               "pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_DCA_used.txt"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    _misc.cprint('ATTENTION: Make sure that ref_pdb is the reference PDB taken after applying a forcefield, since added hyrogen atoms will shift the atom numbers.\n', cfg.cprint_color)

    # read files
    RES, ATOM, NAME = parsePDB_RES_ATOM_NAME(ref_pdb, ref_skiprows)
    DCA_PAIR, _ = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=usecols, skiprows=DCA_skiprows, filter_DCA=filter_DCA)

    RES_PAIR = []
    ATOM_PAIR = []

    # warning
    if n_bonds not in ["all", -1, 0, 1, 2, 3]:
        print("""DCAREX_res2atom warning: invalid n_bonds parameter. Valid values are ["all",-1,0,1,2,3]. Set n_bonds to 1.""")
        n_bonds = 1

    # case1: all atoms
    if n_bonds == "all" or n_bonds == 0 or n_bonds == -1:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            ATOM1 = ATOM[_temp_min:_temp_max]

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            ATOM2 = ATOM[_temp_min:_temp_max]
            RES_PAIR.append((IJ[0], IJ[1]))

            TEMP = []
            for item1 in ATOM1:
                for item2 in ATOM2:
                    TEMP.append((item1, item2))
                ATOM_PAIR.append(TEMP)

    # case2: CA atoms
    elif n_bonds == 1:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append((ATOM[CA1_index], ATOM[CA2_index]))

    # case3: CA and CB atoms
    elif n_bonds == 2:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append([(ATOM[CA1_index], ATOM[CA2_index]),
                              (ATOM[CB1_index], ATOM[CB2_index])])
    # case4: CA, CB and CG atoms
    elif n_bonds == 3:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB1_index = _temp_min + _temp_index
            if "CG" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG')
                CG1_index = _temp_min + _temp_index
            elif "CG1" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG1')
                CG1_index = _temp_min + _temp_index
            elif "CG2" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG2')
                CG1_index = _temp_min + _temp_index
            elif "C"in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('C')
                CG1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB2_index = _temp_min + _temp_index
            if "CG" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG')
                CG2_index = _temp_min + _temp_index
            elif "CG1" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG1')
                CG2_index = _temp_min + _temp_index
            elif "CG2" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG2')
                CG2_index = _temp_min + _temp_index
            elif "C" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('C')
                CG2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append([(ATOM[CA1_index], ATOM[CA2_index]),
                              (ATOM[CB1_index], ATOM[CB2_index]),
                              (ATOM[CG1_index], ATOM[CG2_index])])
    if save_log:
        if "PDBID" in cfg.save_as:
            if cfg.pdbid == "auto":
                cfg.pdbid = _misc.get_PDBid(ref_pdb)
            new_dir = _misc.mkdir(cfg.default_dir)
            cfg.save_as = f"{new_dir}/{cfg.pdbid.upper()}_DCA_used.txt"
        print("Saved log as:", _misc.realpath(cfg.save_as))
        with open(cfg.save_as, "w") as fout:
            if n_bonds == 1:
                fout.write("#DCA contacts top{} ({} bond per contact)\n".format(n_DCA, n_bonds))
            elif n_bonds in [2, 3]:
                fout.write("#DCA contacts top{} ({} bonds per contact)\n".format(n_DCA, n_bonds))
            else:
                fout.write("#DCA contacts top{} (all heavy atom bonds per conact)\n".format(n_DCA))
            fout.write("{}\t{}\t{}\t{}\n".format("#RESi", "RESj", "ATOMi", "ATOMj"))
            for i in range(len(RES_PAIR)):
                if type(ATOM_PAIR[i]) == tuple:
                    fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], ATOM_PAIR[i][0], ATOM_PAIR[i][1]))
                else:
                    for item in ATOM_PAIR[i]:
                        fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], item[0], item[1]))
    return RES_PAIR, ATOM_PAIR


def DCAREX_modify_scoreFile(score_fin, shift_res, res_cols=(2, 3), score_col=(5), **kwargs):
    """
    Modify score file (MSA scores) by shifting residues.

    Args:
        score_fin (str): path to score file
        shift_res (int): shift residues by this value
        outputFileName (str): realpath to modified score file
            if "PDBID_mod.score" is left as default: try to automatically detect
            pattern based on score_fin and add the "_mod" part into filename.
        res_cols (tuple/list): columns with residue numbers
        score_col (tuple/list): column with score/confidence

    Kwargs:
        save_as (str): "PDBID_mod.score"
            if "PDBID_mod.score" (default): try to automatically detect pattern
            based on score_fin and insert the "_mod" part into filename.

    Returns:
        save_as (str): realpath to modified score file
    """
    default = {"save_as": "PDBID_mod.score"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    resi, resj = _misc.read_file(score_fin, usecols=res_cols, skiprows='auto', dtype=np.int_)
    score = _misc.read_file(score_fin, usecols=score_col, skiprows='auto', dtype=np.float_)

    # shift residues
    print(f"Shifting score file residues by {shift_res}.")
    resi += shift_res
    resj += shift_res

    if cfg.save_as == "PDBID_mod.score":
        dirpath = _misc.dirpath(score_fin)
        base = _misc.get_base(score_fin)
        ext = _misc.get_extension(score_fin)
        cfg.save_as = f"{dirpath}/{base}_mod{ext}"

    cfg.save_as = _misc.realpath(cfg.save_as)
    with open(cfg.save_as, "w") as fout:
        for i in range(len(resi)):
            fout.write(f"{resi[i]}\t{resj[i]}\t{score[i]}\n")

    print(f"Saved file as: {cfg.save_as}")
    return cfg.save_as


def DCAREX_modify_topology(top_fin, DCA_used_fin, force_k=10, DCA_skiprows="auto", **kwargs):
    """
    Modify topology:
        - top_fin (topol.top file): use as template
        - DCA_used_fin (DCA_used.txt): use all contacts as restraints
        - modify bond section of new topology by adding contacts with constant force constant

    Args:
        top_fin (str): topology file (path)
        DCA_used_fin (str): DCA used file (path)
        force_k (int/float): force constant of DCA pairs
        DCA_skiprows (int): ignore header rows of DCA_used_fin
            -1 or "auto": auto detect

    Kwargs:
        pdbid (str):
            "auto" (default): detect PDBID based on ref_PDB path
        default_dir (str): "./"
        save_as (str): "PDBID_topol_mod.top"
            detect and replace PDBID in save_as based on ref_PDB path
            if kwarg <pdbid> is "auto" (default).
    """
    default = {"pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_topol_mod.top"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # DCA_used_fin has 4 cols with RESi, RESj, ATOMi, ATOMj -> usecols=(2,3)
    ATOM_I, ATOM_J = _misc.read_file(fin=DCA_used_fin, usecols=(2, 3), skiprows=DCA_skiprows, dtype=np.int_)

    # get n_DCA from DCA_used_fin
    with open(DCA_used_fin, "r") as read:
        WORDS = read.readline().split()
        for item in WORDS:
            if "top" in item:
                n_DCA = item[3:]

    # lstrip ./ or / from save_as
    if "/" in cfg.save_as:
        cfg.save_as = cfg.save_as.lstrip("/")
        cfg.save_as = cfg.save_as.lstrip("./")
    if cfg.pdbid == "auto":
        cfg.pdbid = _misc.get_PDBid(DCA_used_fin)
    if "PDBID" in cfg.save_as:
        cfg.save_as = f"{cfg.pdbid}_topol_mod.top"

    # read and write files
    new_dir = _misc.mkdir(cfg.default_dir)
    output = f"{new_dir}/{cfg.save_as}"
    with open(top_fin, "r") as fin, open(output, "w") as fout:

        for line in fin:
            # default case: copy paste topology file
            fout.write(line)

            # add DCA bonds into bonds section
            if "[ bonds ]" in line:
                fout.write(f"; DCA bonds top{n_DCA}\n")
                fout.write(";  ai    aj funct            c0            c1            c2            c3\n")
                for i in range(len(ATOM_I)):
                    if type(force_k) is int:
                        fout.write("{:5d} {:5d} {:5d} {:13d} {:13d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, force_k))
                    elif type(force_k) is float or str:
                        if _misc.get_float_precision(force_k) == -1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, int(force_k)))
                        if _misc.get_float_precision(force_k) == 1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.1f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(force_k)))
                        else:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.2f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(force_k)))
                fout.write("; native bonds\n")

    print("Saved modified topology as:", output)
    return


### TODO
def DCAREX_modify_topology_v2(top_fin, temp_fin, force_range=[10, 40], lambda_scaling="T", **kwargs):
    """
    TODO

    Modify topology with lambda scaling prop to temperature.

    Args:
        top_fin
        temps_fin
        lambda_scaling:
            T: prop. to T
          1/T: prop to 1/T

    Kwargs:
        pdbid (str):
            "auto" (default): detect PDBID based on ref_PDB path
        default_dir (str): "./"
        save_as (str) : "PDBID_topol_mod.top"
    """
    default = {"pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_topol_mod.top"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # 1) calculate lambdas prop/antiprop to temperatures
    with open(temp_fin, "r") as fin:
        for line in fin:
            TEMPS = line
        TEMPS = TEMPS.split(", ")
        TEMPS = [float(item) for item in TEMPS[:-1]]

        T_min = TEMPS[0]
        T_max = TEMPS[-1]

        LAMBDA = []
        for T in TEMPS:
            lambda_i = (T-T_min)/(T_max-T_min)
            LAMBDA.append(round(lambda_i, 3))
        if lambda_scaling == "1/T":
            LAMBDA = [(1-lambda_i) for lambda_i in LAMBDA]

        FORCE = []
        force_min = force_range[0]
        force_max = force_range[-1]
        force_delta = force_max - force_min
        for lambda_i in LAMBDA:
            FORCE.append(force_min + lambda_i*force_delta)

    # 2) write modified topology for each replica with scaling lambda
    # lstrip ./ or / from save_as
    if "/" in cfg.save_as:
        cfg.save_as = cfg.save_as.lstrip("/")
        cfg.save_as = cfg.save_as.lstrip("./")
    if "PDBID" in cfg.save_as:
        if cfg.pdbid == "auto":
            cfg.pdbid = _misc.get_PDBid(top_fin)
        new_dir = _misc.mkdir(cfg.default_dir)

    rex_label = 0
    for force in FORCE:
        rex_label += 1
        # read and write files
        cfg.save_as = f"{cfg.pdbid}_topol_mod_rex{rex_label}.top"
        output = f"{new_dir}/{cfg.save_as}"
        with open(top_fin, "r") as fin, open(output, "w") as fout:

            modify_section = False
            for line in fin:
                # find DCA bonds section
                if "; DCA bonds" in line:
                    modify_section = True
                elif "; native bonds" in line:
                    modify_section = False
                    print("Saved modified topology as:", output)

                if modify_section == False:
                    fout.write(line)  # default case: copy paste
                else:
                    if ";" in line:
                        fout.write(line)
                    elif ";" not in line:
                        l = line.split()
                        l = [int(item) for item in l]
                        fout.write("{:5d} {:5d} {:5d} {:13d} {:13.3f}\n".format(l[0], l[1], l[2], l[3], force))

    return


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
        _misc.cprint(f"Saved log as: {_misc.realpath('rex_temps.log')}", "blue")
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
        _misc.cprint(f"Saved log as: {_misc.realpath('rex_temps.log')}", "blue")
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


def prep_REX_mdp(main_dir="./", n_REX=None):
    """
    Prepare REX mdp.

    Args:
        main_dir (str): main directory with rex_1, rex_2, etc.
        n_REX (None/int): number of replica
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
    print("Using rex.mdp as template and changing temperatures according to rex_temps.log...")

    for i in range(1, n_REX+1):
        file_path = _misc.mkdir("rex_" + str(i))
        with open(_misc.joinpath(main_dir, "rex.mdp"), "r") as fin, open(file_path + "/rex.mdp", "w") as fout:
            print("Saved mdp file as: " + file_path + "/rex.mdp")

            for line in fin:
                if "ref_t" in line:
                    fout.write("ref_t   = %s       %s         ; reference temperature, one for each group, in K\n" % (REX_TEMPS[i-1], REX_TEMPS[i-1]))
                elif "gen_temp" in line:
                    fout.write("gen_temp = %s  ; temperature for Maxwell distribution\n" % (REX_TEMPS[i-1]))
                else:
                    fout.write(line)

    return
