# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: rna.py
# @Last modified by:   arthur
# @Last modified time: 19.05.2021


"""
This module contains functions modify RNA/DNA topologies to include contact bias etc.
"""

import pyrexMD.misc as _misc
import MDAnalysis as mda


def nucleic_parsePDB(fin, sel="name N1 or name N3", filter=True):
    """
    Reads PDB file and returns the columns for residue id, residue name,
    atom id and atom name as lists.

    Args:
        fin (str): PDB file
        sel (str): selection str
        filter (bool): True ~ return only N1 atoms for A,G and N3 atoms for T,C,U

    Returns:
        RESID (list)
            residue id column of PDB
        RESNAME (list)
            residue name column of PDB
        ID (list)
            atom id column of PDB
        NAME (list)
            atom name column of PDB
    """
    map = {"A": "N1",
           "ADE": "N1",
           "G": "N1",
           "GUA": "N1",
           "T": "N3",
           "THY": "N3",
           "C": "N3",
           "CYT": "N3",
           "U": "N3",
           "URA": "N3"}
    ############################################################################
    RESID = []
    RESNAME = []
    ID = []
    NAME = []

    u = mda.Universe(fin)
    a = u.select_atoms(sel)

    for item in a:
        if filter is True:
            if item.name == map[item.resname]:
                RESID.append(item.resid)
                RESNAME.append(item.resname)
                ID.append(item.id)
                NAME.append(item.name)

        else:
            RESID.append(item.resid)
            RESNAME.append(item.resname)
            ID.append(item.id)
            NAME.append(item.name)

    return RESID, RESNAME, ID, NAME


def nucleic_DCA_res2atom_mapping(ref_pdb, DCA_fin, n_DCA, usecols, DCA_skiprows="auto",
                                 filter_DCA=True, save_log=True, **kwargs):
    """
    Get RNA/DNA contact mapping. Uses either N1 or N3 atoms for contact based on
    nucleotide. Returns lists of matching RES pairs and ATOM pairs for contacts
    specified in DCA_fin file.

    Args:
        ref_pdb (str): reference PDB (path)
        DCA_fin (str): DCA file (path)
        n_DCA (int): number of DCA contacts which should be used
        usecols (tuple, list): columns containing the RES pairs in DCA_fin
        DCA_skiprows (int):
          | ignore header rows of DCA_fin
          | -1 or "auto": auto detect
        filter_DCA (bool):
          | True: ignore DCA pairs with abs(i-j) < 3
          | False: use all DCA pairs w/o applying filter
        save_log (bool)

    Keyword Args:
        cprint_color (None, str): colored print color
        pdbid (str): "auto" (default): detect PDBID based on ref_PDB path
        default_dir (str): "./"
        save_as (str):
          | "PDBID_DCA_used.txt"
          | detect and replace PDBID in kwarg <save_as> based on ref_PDB path
          | if kwarg <pdbid> is "auto" (default).

    Returns:
        RES_PAIR (list)
            list with RES pairs
        ATOM_PAIR (list)
            list with ATOM pairs
    """
    default = {"cprint_color": "blue",
               "pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_DCA_used.txt"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    _misc.cprint('ATTENTION: Make sure that ref_pdb is the reference PDB taken after applying a forcefield, since added atoms will shift the atom numbers.\n', cfg.cprint_color)

    # read pdb file and filter for mapping atoms (hardcoded for sel = "name N1 or name N3")
    RESID, RESNAME, ID, NAME = nucleic_parsePDB(ref_pdb, sel="name N1 or name N3")

    # read contact pairs
    DCA_PAIR, _ = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=usecols, skiprows=DCA_skiprows, filter_DCA=filter_DCA)

    RES_PAIR = []
    ATOM_PAIR = []

    for IJ in DCA_PAIR:
        RES_PAIR.append((IJ[0], IJ[1]))
        ATOM_PAIR.append((ID[RESID.index(IJ[0])], ID[RESID.index(IJ[1])]))

    if save_log:
        if "PDBID" in cfg.save_as:
            if cfg.pdbid == "auto":
                cfg.pdbid = _misc.get_PDBid(ref_pdb)
            new_dir = _misc.mkdir(cfg.default_dir)
            cfg.save_as = f"{new_dir}/{cfg.pdbid.upper()}_DCA_used.txt"
        print("Saved log as:", _misc.realpath(cfg.save_as))
        with open(cfg.save_as, "w") as fout:
            fout.write(f"#DCA contacts top{n_DCA} (1 bond per contact)\n")
            fout.write("{}\t{}\t{}\t{}\n".format("#RESi", "RESj", "ATOMi", "ATOMj"))
            for i in range(len(RES_PAIR)):
                if type(ATOM_PAIR[i]) == tuple:
                    fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], ATOM_PAIR[i][0], ATOM_PAIR[i][1]))
                else:
                    for item in ATOM_PAIR[i]:
                        fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], item[0], item[1]))
    return RES_PAIR, ATOM_PAIR


def nucleic_modify_topology(top_fin, DCA_fin, force_k=2, skiprows="auto", **kwargs):
    """
    Modifies topology by using the contacts written in <DCA_fin> file.

    Modify topology:

      - top_fin (topol.top file): use as template
      - DCA_fin: use all contacts as restraints
      - modify bond section of new topology by adding contacts with force constant

    Args:
        top_fin (str): topology file (path)
        DCA_fin (str): DCA file (path)
        force_k (int, float): force constant of contact pairs
        skiprows (int):
          | ignore header rows of DCA_used_fin
          | -1 or "auto": auto detect

    Keyword Args:
        pdbid (str): "auto" (default): detect PDBID based on ref_PDB path
        default_dir (str): "./" (default)
        save_as (str):
          | "PDBID_topol_mod.top"
          | detect and replace PDBID in save_as based on ref_PDB path
          | if kwarg <pdbid> is "auto" (default).
    """
    default = {"pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_topol_mod.top"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    # DCA_used_fin has 4 cols with RESi, RESj, ATOMi, ATOMj -> usecols=(2,3)
    ATOM_I, ATOM_J = _misc.read_file(fin=DCA_fin, usecols=(2, 3), skiprows=skiprows, dtype=int)

    # get n_DCA from DCA_used_fin
    with open(DCA_fin, "r") as read:
        WORDS = read.readline().split()
        for item in WORDS:
            if "top" in item:
                n_DCA = item[3:]

    # lstrip ./ or / from save_as
    if "/" in cfg.save_as:
        cfg.save_as = cfg.save_as.lstrip("/")
        cfg.save_as = cfg.save_as.lstrip("./")
    if cfg.pdbid == "auto":
        cfg.pdbid = _misc.get_PDBid(DCA_fin)
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
