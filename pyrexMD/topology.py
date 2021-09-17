# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: topology.py
# @Last modified by:   arthur
# @Last modified time: 17.09.2021


"""
.. hint :: This module contains functions for modifying universe topologies,
    e.g., align atoms/residues of two universes, get matching selection strings,
    include bias contacts.

"""

import pyrexMD.misc as _misc
import MDAnalysis as mda
import numpy as np
import os
from tqdm.notebook import tqdm


################################################################################
################################################################################
### Modify universe / topology


def get_resids_shift(mobile, ref):
    """
    Compares universe.residues.resnames between mobile and ref in order to get
    resids shift.

    Args:
        mobile (universe)
        ref (universe)

    Returns:
        shift (int)
            shift value of ref residues to match mobile residues
    """
    # compare resnames to get start_ndx
    start_ndx = _misc.get_subarray_start_ndx(mobile.residues.resnames, ref.residues.resnames)

    # start_ndx is static, but shift value should be either start_ndx or 0 (after previous shift)
    # test if shift already performed
    if min(ref.residues.resids) == 1:
        shift = start_ndx
    else:
        shift = 0

    # test if resnames match (due to same length) but resids are shifted
    if len(mobile.residues.resids) == len(ref.residues.resids):
        diff = min(mobile.residues.resids)-min(ref.residues.resids)
        if diff != shift:
            shift = diff

    return shift


def shift_resids(u, shift=None, verbose=True):
    """
    Shift mda.universe.residues by <shift> value.

    Args:
        u (universe)
        shift (None, int): shift value
        verbose (bool)
    """
    if isinstance(shift, (np.ndarray, list)) and np.all(shift == shift[0]):
        shift = shift[0]
    if shift == None or shift == 0:
        return

    for item in u._topology.attrs:
        if isinstance(item, mda.core.topologyattrs.Resids):
            before = u.residues.resids
            item.set_residues(u.residues, item.get_residues(u.residues)+shift)
            after = u.residues.resids
        if isinstance(item, mda.core.topologyattrs.Resnums):
            item.set_residues(u.residues, item.get_residues(u.residues)+shift)

    if verbose:
        print(f"Shifting resids from:\n{before}")
        print(f"\nto:\n{after}")
    return


def align_resids(mobile, ref, norm=True, verbose=True, **kwargs):
    """
    Align resids of mobile and ref by comparing universe.residues.resnames and
    shifting the resids of reference (usually smaller than mobile).

    Args:
        mobile (universe)
        ref (universe)
        norm (bool): apply topology.norm_resids()
        verbose (bool)

    Keyword Args:
        cprint_color (None, str): colored print color
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if norm:
        norm_resids(mobile, 'mobile', verbose=verbose)
        norm_resids(ref, 'reference', verbose=verbose)

    shift = get_resids_shift(mobile, ref)
    if shift != 0:
        _misc.cprint("Aligning reference res ids...\n", cfg.cprint_color)
        shift_resids(ref, shift, verbose=verbose)
    return


def __HELP_print(info='', verbose=True):
    if verbose:
        if info == '':
            print('Norming atom ids...')
        else:
            print(f'Norming {info} atom ids...')
    else:
        return


def norm_ids(u, info='', verbose=True):
    """
    Modify existing MDA universe/atomgrp and normalize ids according to
    min(universe.atoms.ids) = 1

    Args:
        u (universe, atomgrp): structure
        info (str):
          | additional info for print message
          | 'reference'
          | 'mobile'
        verbose (bool)

    Example:
        | >> ref = mda.Universe(<top>)
        | >> print(ref.atoms.ids)
        | [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        |
        | >> norm_ids(ref, 'reference')
        | Norming atom ids of reference...
        |
        | >> print(ref.atoms.ids)
        | [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    """
    if not isinstance(u, (mda.core.universe.Universe, mda.core.groups.AtomGroup)):
        raise TypeError('''{norm_ids.__module__}.{norm_ids.__name__}():\
        \nInvalid input. u must be MDA universe/atomgrp.''')

    if min(u.atoms.ids) != 1:
        shift = 1 - min(u.atoms.ids)
        u.atoms.ids += shift
        __HELP_print(info, verbose)
    return


def norm_resids(u, info='', verbose=True):
    """
    Modify existing MDA universe/atomgrp and normalize resids according to
    min(universe.residues.resids) = 1

    Args:
        u (universe, atomgrp): structure
        info (str):
          | additional info for print message
          | 'reference'
          | 'topology'

    Example:
        | >> ref = mda.Universe(<top>)
        | >> print(ref.residues.resids)
        | [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        |
        | >> norm_resids(u, 'reference')
        | Norming resids of reference...
        |
        | >> print(ref.residues.resids)
        | [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    """
    if not isinstance(u, (mda.core.universe.Universe, mda.core.groups.AtomGroup)):
        raise TypeError('''{norm_resids.__module__}.{name_resids.__name__}():\
        \nInvalid input. u must be MDA universe/atomgrp.''')

    if min(u.residues.resids) != 1:
        shift = 1 - min(u.residues.resids)
        shift_resids(u, shift, verbose=False)
        __HELP_print(info, verbose)
    return


def norm_universe(u, *args, info='', verbose=True):
    """
    Executes for each universe:

      - norm_ids(u, info)
      - norm_resids(u, info)

    Args:
        u (universe, atomgrp, list, array): universe or list/array with universes
        info (str):
          | additional info for print message
          | 'reference'
          | 'mobile'
        verbose (bool)
    """
    if isinstance(u, (list, np.ndarray)):
        for item in u:
            norm_ids(item, info, verbose)
            norm_resids(item, info, verbose)
    else:
        norm_ids(u, info, verbose)
        norm_resids(u, info, verbose)
    return


def norm_and_align_universe(mobile, ref, verbose=True):
    """
    - Norm reference and mobile universe
    - Align mobile universe on reference universe (matching atom ids and res ids)

    Args:
        mobile (universe, atomgrp)
        ref    (universe, atomgrp)
        verbose (bool)
    """

    if len(ref.atoms.ids) != len(mobile.atoms.ids):
        raise ValueError(f'''{norm_and_align_universe.__module__}.{norm_and_align_universe.__name__}():\
        \nNumber of atoms doesn't match! Cannot align atom ids.''')

    if min(ref.atoms.ids) < 1:
        norm_ids(ref, 'reference')

    if min(mobile.atoms.ids) < 1:
        norm_ids(mobile, 'mobile')

    if np.any(ref.atoms.ids != mobile.atoms.ids):
        print('Reference and mobile atom ids are inconsistent! Aligning mobile atom ids to match with reference structure...')
        shift = ref.atoms.ids - mobile.atoms.ids
        mobile.atoms.ids += shift

    if len(ref.residues.resids) != len(mobile.residues.resids):
        raise ValueError(f'''{norm_and_align_universe.__module__}.{norm_and_align_universe.__name__}()::\
        \nResidues number doesn't match! Cannot align resids.''')

    if min(ref.residues.resids) < 1:
        norm_resids(ref, 'reference')

    if min(mobile.residues.resids) < 1:
        norm_resids(mobile, 'mobile')

    if np.any(ref.residues.resids != mobile.residues.resids):
        print('Reference and mobile resids are inconsistent! Aligning mobile resids to match with reference structure...')
        shift = ref.residues.resids - mobile.residues.resids
        shift_resids(mobile, shift=shift, verbose=verbose)

    print("Both universes are normed and aligned (atom ids + resids).")
    return


def true_select_atoms(u, sel='protein', ignh=True, norm=True):
    """
    Create "true selection" (atomgroup copy in NEW universe) after applying

      - norm_ids()
      - norm_resids()

    Note that atom indices are reassigned according to range(0, len(u.atoms))

    Args:
        u (str, universe, atomgrp): structure path (PDB id) or universe with structure
        sel (str): selection string
        ignh (bool): ignore hydrogen (mass < 1.2)
        norm (bool): apply topology.norm_universe()

    Returns:
        a (atomgrp)
            "true selection" ~ atomgroup copy in NEW universe

    .. Note:: mda.select_atoms() remembers information about the original
       universe configuration which is sometimes NOT WANTED.

    EXAMPLE:
      | # Init universe
      | >> u = mda.Universe(<top>)
      |
      | # select_atoms() behaviour
      | >> a = u.select_atoms("protein and type C")
      | >> a
      | <AtomGroup with 72 atoms>
      | >> a.residues.atoms
      | <AtomGroup with 964 atoms>
      |
      | # true_select_atoms() behaviour
      | >> a = topology.true_select_atoms(u, sel="protein and type C")
      | >> a
      | <Universe with 72 atoms>
      | >> a.residue.atoms
      | <AtomGroup with 72 atoms>
    """
    # case 1: input is PDB ID -> fetch online
    if type(u) is str and len(u) == 4:
        u = mda.fetch_mmtf(u)
    # case 2: input is path -> create MDA Universe
    elif type(u) is str and len(u) > 4:
        u = mda.Universe(u)
    # case 3: input is MDA Universe -> just use it

    if norm:
        norm_universe(u)
    a0 = u.select_atoms(sel)
    if ignh:
        a0 = u.select_atoms(sel + ' and prop mass > 1.2')
    u = mda.core.universe.Merge(a0)
    a = u.atoms
    return a


def get_matching_selection(mobile, ref, sel="protein and name CA", norm=True, verbose=True):
    """
    Get matching selection strings of mobile and reference after resids alignment.

    Args:
        mobile (universe)
        ref (universe)
        sel (str): selection string
        norm (bool): apply topology.norm_resids()
        verbose (bool)

    Returns:
        sel1 (str)
            matching selection string of mobile
        sel2 (str)
            matching selection string of ref
    """
    if get_resids_shift(mobile, ref) != 0:
        align_resids(mobile, ref, norm=norm, verbose=verbose)
    sel1 = f"{sel} and resid {min(ref.residues.resids)}-{max(ref.residues.resids)}"
    sel2 = f"{sel}"

    if np.all(ref.select_atoms(sel1).residues.resnames == ref.select_atoms(sel2).residues.resnames):
        pass
    else:
        raise ValueError("No matching selection found.")
    return sel1, sel2


def check_matching_selection(mobile, ref, sel="protein", verbose=True):
    """
    Gets matching selection strings for mobile and reference and then tests if
    matching selections have equal atoms and residues. Returns matching
    selection strings.

    Args:
        mobile (universe)
        ref (universe)
        sel (str): selection string
        verbose (bool):
          | True: prints detailed report
          | False: prints only selection strings and if atoms/residues match
    """
    # modify selection string if necessary
    sel1, sel2 = get_matching_selection(mobile, ref, sel=sel)
    _misc.cprint("Matching selection strings:", "blue")
    print(f"Mobile:    sel1 = {sel1}")
    print(f"Reference: sel2 = {sel2}")

    a1 = mobile.select_atoms(sel1)
    a2 = ref.select_atoms(sel2)
    if verbose:
        _misc.cprint("\nAtom counts:", "blue")
        print(f"Mobile:    sel1 contains {a1.n_atoms} of {mobile.atoms.n_atoms} atoms.")
        print(f"Reference: sel2 contains {a2.n_atoms} of {ref.atoms.n_atoms} atoms.")
    if (a1.atoms.n_atoms == a2.atoms.n_atoms) and (mobile.atoms.n_atoms == ref.atoms.n_atoms):
        _misc.cprint("Atom counts match", "green")
    else:
        _misc.cprint("Atom counts do not match", "red")

    if verbose:
        _misc.cprint("\nRES in sel1:", "blue")
        print(a1.residues.resnames)
        _misc.cprint("RES in sel2:", "blue")
        print(a2.residues.resnames)
    if np.all(a1.residues.resnames == a2.residues.resnames):
        _misc.cprint("RES match\n", "green")
    else:
        _misc.cprint("RES do not match\n", "red")
    return sel1, sel2


def check_residues(mobile, ref, sel1="all", sel2="all", verbose=True):
    """
    Compares residues of mobile and reference universe using specific selections.

    Args:
        mobile (universe)
        ref (universe)
        sel1 (str): selection string of mobile
        sel2 (str): selection string of ref
        verbose (bool):
          | True: prints detailed report
          | False: prints only if residue ids and names match
    """
    a1 = mobile.atoms.select_atoms(sel1)
    a2 = ref.atoms.select_atoms(sel2)
    if verbose:
        _misc.cprint("Mobile:", "blue")
        print(a1.residues.resids)
        print(a1.residues.resnames)

        _misc.cprint("\nRef:", "blue")
        print(a2.residues.resids)
        print(a2.residues.resnames)
        print()

    if np.all(a1.residues.resids == a1.residues.resids):
        _misc.cprint("RES ids match", "green")
    else:
        _misc.cprint("RES ids do not match", "red")
    if np.all(a2.residues.resnames == a2.residues.resnames):
        _misc.cprint("RES names match", "green")
    else:
        _misc.cprint("RES names do not match", "red")
    if len(a1.atoms) != len(a2.atoms):
        _misc.cprint("Warning: sel1 and sel2 have unequal atom counts!", "red")
        _misc.cprint(f"Atoms in mobile: {len(a1.atoms)}", "red")
        _misc.cprint(f"Atoms in reference: {len(a2.atoms)}", "red")
    return


def sel2selid(u, sel, norm=True, filter_rna=True):
    """
    Converts selection string to selection id string.

    Args:
        u (universe, str): structure universe or path to structure pdb
        sel (str): selection string
        norm (bool): apply norm_universe()
        filter_rna (bool): filter selection only for N1 and N3 atoms based on nucleic residues

    Returns:
        selid (str)
            selection id string

    Example:
        | >> u = mda.Universe("path/to/pdb")
        | >> selid = sel2selid(u, sel="protein and name CA")
        | >> selid
        | 'index 4 18 37 58 77 94 118 137 159 171 178 193 199 210 221 228 260 274 288 294'
        | >> u.select_atoms("protein and name CA") == u.selectg_atoms(selid)
        | True
    """
    # filter_rna mapping selection
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
    ########################################################
    if isinstance(u, str):
        u = mda.Universe(u)
    if norm:
        norm_universe(u)
    a = u.select_atoms(sel)

    for item in a:
        if filter_rna is True and len(a.select_atoms("nucleic")) != 0:
            if item.name != map[item.resname]:
                a = a - item    # remove atoms which are not in map

    ids = a.atoms.ids-1
    ids = [str(item) for item in ids]
    selid = "index " + " ".join(ids)
    return selid


def dump_structure(u, frames, save_as, default_dir="./structures", sel="protein"):
    """
    Dump structures for a list of frames with the file name "save_as".
    Automatically prepends frame number to the extension.

    Args:
        u (universe, atomgrp): universe containing the structure
        frames (array, list)
        save_as (str): save name or realpath to save file. frame number will
          be prepended to the extension
        default_dir (str)
        sel (str): selection string

    Returns:
        dirpath (str)
            realpath to directory with dumped structures
    """
    if "." in save_as:
        realpath = _misc.joinpath(default_dir, save_as, create_dir=True)
        dirpath = os.path.dirname(os.path.realpath(realpath))
        base = _misc.get_base(realpath)
        ext = _misc.get_extension(realpath)

    else:
        raise _misc.Error("specify a file format within the 'save_as' string!")

    # save structures
    for ts in tqdm(u.trajectory[frames]):
        frame = u.trajectory.frame
        structure = u.select_atoms(sel)
        temp_save_as = f"{dirpath}/{base}_{frame}{ext}"
        structure.write(temp_save_as)
    print("Dumped structures into:", dirpath)
    return dirpath


################################################################################
################################################################################
### Include contact bias

def parsePDB(fin, sel, norm=False, filter_rna=True, verbose=False):
    """
    Reads PDB file and returns the columns for residue id, residue name,
    atom id and atom name as lists.

    Args:
        fin (str): PDB file
        sel (str): selection string
        norm (bool): apply norm_universe() before parsing.
        filter_rna (bool): filter selection only for N1 and N3 atoms based on nucleic residues
        verbose (bool)

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
    # filter_rna mapping selection
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
    RESID, RESNAME, ID, NAME = [], [], [], []

    u = mda.Universe(fin)
    if norm:
        norm_universe(u, verbose=verbose)
    a = u.select_atoms(sel)

    for item in a:
        if filter_rna is True and len(a.select_atoms("nucleic")) != 0:
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


def DCA_res2atom_mapping(ref_pdb, DCA_fin, n_DCA, usecols, DCA_skiprows="auto",
                         filter_DCA=True, save_log=True, **kwargs):
    """
    Get contact mapping. Returns lists of matching RES pairs and ATOM pairs for contacts
    specified in DCA_fin file.

    .. Note::
      - maps only CA atoms if ref_pdb is protein.
      - maps only N1 or N3 atoms if ref_pdb is nucleic.

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
        save_log (bool): create log file with contact mapping

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
            list with RES pair tuples (RESi, RESj)
        ATOM_PAIR (list)
            list with ATOM pair tuples (ATOMi, ATOMj)
    """
    default = {"cprint_color": "blue",
               "pdbid": "auto",
               "default_dir": "./",
               "save_as": "PDBID_DCA_used.txt"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    _misc.cprint('ATTENTION: Make sure that ref_pdb is the reference PDB taken after applying a forcefield, since added atoms will shift the atom numbers.\n', cfg.cprint_color)

    # check if ref_pdb is protein or nucleic and apply hardcoded mapping
    ref = mda.Universe(ref_pdb)
    if len(ref.select_atoms("nucleic")) == 0:
        RESID, RESNAME, ID, NAME = parsePDB(ref_pdb, sel="protein and name CA")
    else:
        RESID, RESNAME, ID, NAME = parsePDB(ref_pdb, sel="name N1 or name N3")

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
        print("Saved log as:", os.path.realpath(cfg.save_as))
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


def DCA_modify_topology(top_fin, DCA_used_fin, n_DCA=None, k=10, skiprows="auto", **kwargs):
    """
    Modifies topology by using the contacts written in "DCA_used.txt" file.
    "DCA_used.txt" is supposed to have 4 columns: RESi, RESj, ATOMi, ATOMj.

    Modify topology:

      - top_fin (topol.top file): use as template
      - DCA_used_fin (DCA_used.txt): use all contacts as restraints
      - modify bond section of new topology by adding contacts with force constant

    Args:
        top_fin (str): topology file (path)
        DCA_used_fin (str): DCA file (path)
        n_DCA (None, int):
          | number of used DCA contacts
          | None: search header of `DCA_used_fin` for entry with topn_DCA
        k (int, float): force coefficient of contact pairs
        skiprows (int):
          | ignore header rows of DCA_used_fin
          | -1 or "auto": auto detect

    Keyword Args:
        pdbid (str): "auto" (default): detect PDBID based on ref_PDB path
        save_as (str):
          | "PDBID_topol_mod.top" (default)
          | detect and replace PDBID in save_as based on ref_PDB path
          | if kwarg <pdbid> is "auto" (default).
    """
    default = {"pdbid": "auto",
               "save_as": "PDBID_topol_mod.top"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    with open(DCA_used_fin, "r") as read:
        # if n_DCA is None: use n_DCA value from header of DCA_used_fin
        if n_DCA is None:
            WORDS = read.readline().split()
            for item in WORDS:
                if "top" in item:
                    n_DCA = int(item[3:])
    # DCA_used_fin has 4 cols with RESi, RESj, ATOMi, ATOMj -> usecols=(2,3)
    ATOM_I, ATOM_J = _misc.read_file(fin=DCA_used_fin, usecols=(2, 3),
                                     n_rows=n_DCA, skiprows=skiprows, dtype=int)
    if n_DCA is None:
        n_DCA = len(ATOM_I)

    if "PDBID" in cfg.save_as and cfg.pdbid == "auto":
        cfg.pdbid = _misc.get_PDBid(DCA_used_fin)
        cfg.save_as = f"{cfg.pdbid}_topol_mod.top"

    # read and write files
    with open(top_fin, "r") as fin, open(cfg.save_as, "w") as fout:

        for line in fin:
            # default case: copy paste topology file
            fout.write(line)

            # add DCA bonds into bonds section
            if "[ bonds ]" in line:
                fout.write(f"; DCA bonds top{n_DCA}\n")
                fout.write(";  ai    aj funct            c0            c1            c2            c3\n")
                for i in range(len(ATOM_I)):
                    if type(k) is int:
                        fout.write("{:5d} {:5d} {:5d} {:13d} {:13d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, k))
                    elif type(k) is float or str:
                        if _misc.get_float_precision(k) == -1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, int(k)))
                        if _misc.get_float_precision(k) == 1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.1f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(k)))
                        else:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.2f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(k)))
                fout.write("; native bonds\n")

    print("Saved modified topology as:", os.path.realpath(cfg.save_as))
    return


def DCA_modify_scoreFile(score_fin, shift_res, res_cols=(0, 1), score_col=(2), **kwargs):
    """
    Modify score file (MSA scores) by shifting residues.

    Args:
        score_fin (str): path to score file
        shift_res (int): shift residues by this value
        outputFileName (str):
          | realpath to modified score file.
          | if "PDBID_mod.score" is left as default: try to automatically detect
          | pattern based on score_fin and add the "_mod" part into filename.
        res_cols (tuple/list): score_fin columns with residue numbers
        score_col (tuple/list): score_fin column with score/confidence

    Keyword Args:
        save_as (str):
          | "PDBID_mod.score"
          | if "PDBID_mod.score" (default): try to automatically detect pattern
          | based on score_fin and insert the "_mod" part into filename.

    Returns:
        save_as (str)
            realpath to modified score file
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
        dirpath = os.path.dirname(os.path.realpath(score_fin))
        base = _misc.get_base(score_fin)
        ext = _misc.get_extension(score_fin)
        cfg.save_as = f"{dirpath}/{base}_mod{ext}"

    cfg.save_as = os.path.realpath(cfg.save_as)
    with open(cfg.save_as, "w") as fout:
        for i in range(len(resi)):
            fout.write(f"{resi[i]}\t{resj[i]}\t{score[i]}\n")

    print(f"Saved file as: {cfg.save_as}")
    return cfg.save_as
