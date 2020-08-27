# TODO: instead of creating new universes in each function: check if input is str or universe
#       if str: create new universe
#       if universe: use universe
# TODO: class Universe(object) -> Inherit mda.Universe class and make ana.functions accessible as properties.


from __future__ import division, print_function
import myPKG.misc as _misc
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
from MDAnalysis.analysis import distances as _distances, rms as _rms, align as _align
import MDAnalysis as mda
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import operator
import os
import glob


# global update for plots
# matplotlib.rcParams.update({'font.family': "sans-serif", 'font.weight': "normal", 'font.size': 16})
cp = sns.color_palette()  # access seaborn colors via cp[0] - cp[9]


################################################################################
################################################################################
# TODO: FEATURES


class Universe(object):
    """
    TODO

    Inherit mda.Universe class and make ana.functions accessible as properties.

    Example:
    u = ana.Universe("topol.top")
    u.get_Native_Contacts
    u.plot_Contact_Map

    """

    def __init__(self, ref='', top='', traj=''):
        self.ref = mda.Universe(ref)
        if top != '' and traj != '':
            self = mda.Universe(top, traj)
        elif top != '':
            self = mda.Universe(top)
        elif traj != '':
            self = mda.Universe(traj)

        # TODO: make ana.functions as properties accessible to object
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        return


################################################################################
################################################################################
# alias functions / code from MDAnalysis


# def rmsd(mobile, ref, sel='backbone', superposition=True):
#     """
#     Calculates the rmsd between mobile and ref.
#
#     Alias function of MDAnalysis.analysis.rms.rmsd. See help(MDAnalysis.analysis.rms.rmsd) for more information
#
#     Args:
#         mobile (MDA universe/atomgrp): mobile structure
#         ref (MDA universe/atomgrp): reference structure
#         mobile (np.array): mobile structure atom positions
#         ref (np.array): reference structure atom positions
#         sel (str): selection string, only applies if mobile and ref are either MDA universe or atomgrp
#         superposition (bool):
#             True: calculate rmsd after translation + rotation
#             False: calculate rmsd w/o translation + rotation
#
#     Returns:
#         rmsd (float): rmsd between mobile and ref wrt. sel
#     """
#     if type(mobile).__module__ == np.__name__ and \
#             type(ref).__module__ == np.__name__:
#         rmsd = _rms.rmsd(mobile, ref, superposition=superposition)
#     else:
#         mobile = mobile.select_atoms(sel).positions
#         ref = ref.select_atoms(sel).positions
#         rmsd = _rms.rmsd(mobile, ref, superposition=superposition)
#     return rmsd

def get_time_conversion(u):
    """
    get/print time conversion of MDA universe <u>

    Args:
        u (MDA universe)
    """
    if isinstance(u, mda.core.universe.Universe):
        dt = u.trajectory.dt
        tu = u.trajectory.units["time"]
        if tu == "ps":
            print(f"Time = Frame * {dt} {tu} = Frame * {0.001*dt} ns")
        else:
            print(f"Time=Frame * {dt} {tu}")
    else:
        raise TypeError("type(u) must be MDAnalysis.core.universe.Universe.")
    return


def get_RMSD(mobile, ref, sel1='backbone', sel2='backbone', weights='mass', plot=False, alt_return=True, **kwargs):
    """
    Calculates the RMSD between mobile and ref after superimposing (translation and rotation).
    Alias function of MDAnalysis.analysis.rms.RMSD. See help(MDAnalysis.analysis.rms.RMSD) for more information.

    Args:
        mobile (MDA universe/atomgrp): mobile structure
        ref (MDA universe/atomgrp): reference structure
        sel1 (str): selection string of mobile structure
        sel2 (str): selection string of reference structure
        weights (bool/str/array_like): weights during superposition of mobile and reference
           None: no weights
           "mass": atom masses as weights
           array_like: If a float array of the same length as "mobile" is provided, use
                       each element of the "array_like" as a weight for the corresponding
                       atom in "mobile".

        plot (bool): create plot
        alt_return (bool): alternative return datatype
            True:
                ftr_tuple = RMSD()    # tuple of arrays with shape (length, )
                                      # ftr: frame, time, rmsd
                frame= ftr_tuple[0]
                time = ftr_tuple[1]
                rmsd = ftr_tuple[2]

            False:
                ftr_array = get_RMSD()    # single array  with shape (length, 3)
                                          # ftr: frame, time, rmsd
                frame= ftr_array[:,0]
                time = ftr_array[:,1]
                rmsd = ftr_array[:,2]

    Kwargs:
        see help(misc.figure)

    Returns:
        if alt_return == True:
            ftr_tuple (tuple)    # ftr: frame, time, rmsd
        if alt_return == False:
            ftr_array (array)    # ftr: frame, time, rmsd
    """
    if not (isinstance(mobile, (mda.core.universe.Universe, mda.core.groups.AtomGroup))
            and isinstance(ref, (mda.core.universe.Universe, mda.core.groups.AtomGroup))):
        raise TypeError(f'''{get_RMSD.__module__}.{get_RMSD.__name__}():\
        \nmobile and ref must be MDA universe or atomgrp!''')
        return

    else:
        RMSD = _rms.RMSD(mobile.select_atoms(sel1), ref.select_atoms(sel2), weights=weights)
        RMSD.run()

        ftr_array = RMSD.rmsd
        frame = RMSD.rmsd[:, 0]
        time = RMSD.rmsd[:, 1]
        rmsd = RMSD.rmsd[:, 2]

    if plot:
        #PLOT(xdata=frame, ydata=rmsd, xlabel='Frame', ylabel=f'RMSD ($\AA$)', **kwargs)
        PLOT(xdata=time, ydata=rmsd, xlabel='Time (ps)', ylabel=f'RMSD ($\AA$)', **kwargs)

    if alt_return is False:
        return ftr_array

    elif alt_return is True:
        return frame, time, rmsd


def get_RMSF(mobile, sel='protein and name CA', plot=False):
    """
    Calculates the RMSF of mobile.
    Alias function of MDAnalysis.analysis.rms.RMSF.
    See help(MDAnalysis.analysis.rms.get_RMSF) for more information.

    Args:
        mobile (MDA universe/atomgrp): mobile structure
        sel (str): selection string
        plot (bool): create plot

    Returns:
        RMSF (array)
    """
    if not isinstance(mobile, (mda.core.universe.Universe, mda.core.groups.AtomGroup)):
        raise TypeError(f'''{get_RMSF.__module__}.{get_RMSF.__name__}():\
        \nmobile be MDA universe or atomgrp!''')
        return

    else:
        RMSF = _rms.RMSF(mobile.select_atoms(sel))
        RMSF.run()
        rmsf = RMSF.rmsf

    if plot:
        if 'name CA' in sel:
            PLOT(xdata=mobile.select_atoms(sel).resids, ydata=rmsf,
                 xlabel='RESID', ylabel='RMSF')

        else:
            PLOT(xdata=mobile.select_atoms(sel).ids, ydata=rmsf,
                 xlabel='ID', ylabel='RMSF')

    return rmsf


def get_Distance_Matrices(mobile, sss=[None, None, None],
                          sel="protein and name CA", flatten=False,
                          **kwargs):
    """
    Calculate distance matrices for mobile and return them.

    Args:
        mobile (MDA universe/atomgrp): structure with trajectory
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        sel (str): selection string
        flatten (bool): returns flattened distance matrices

    Kwargs:
        aliases for sss items:
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size

    Returns:
        DM (np.array): array of distance matrices
    """
    ############################################################################
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]
               }
    cfg = _misc.CONFIG(default, **kwargs)
    cfg = _HELP_sss_None2int(mobile, cfg)  # convert None values of they keys sss, start, stop, step to integers
    ############################################################################
    a = mobile.select_atoms(sel)

    DM = np.empty((len(mobile.trajectory[cfg.start:cfg.stop:cfg.step]),
                   a.n_atoms*a.n_atoms))  # tuple args: length, size (of flattened array)
    for i, ts in enumerate(tqdm(mobile.trajectory[cfg.start:cfg.stop:cfg.step])):
        DM[i] = mda.analysis.distances.distance_array(a.positions, a.positions).flatten()

    if not flatten:
        DM = DM.reshape((len(mobile.trajectory[cfg.start:cfg.stop:cfg.step]),
                         a.n_atoms, a.n_atoms))  # tuple args: length, N_CA, N_CA
    return DM
################################################################################
################################################################################
### analysis.py "core" functions


def norm_ids(u, info=''):
    """
    Modify existing MDA universe/atomgrp and normalize ids according to:
       - min(universe.atoms.ids) > 0
       - min(universe.atoms.ids) can but must not be 1

    Args:
        u (MDA universe/atomgrp): structure
        info (str): additional info for print message
            - 'reference'
            - 'topology'

    Example:
        u = mda.Universe(<top>)
        print(u.atoms.ids)
        >> [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

        norm_ids(u, 'topology')
        >> Norming atom ids of topology...

        print(u.atoms.ids)
        >> [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    """
    if not isinstance(u, (mda.core.universe.Universe, mda.core.groups.AtomGroup)):
        raise TypeError(f'''{norm_ids.__module__}.{norm_ids.__name__}():\
        \nInvalid input. u must be MDA universe/atomgrp.''')
        return

    if min(u.atoms.ids) < 1:
        if info == '':
            print('Norming atom ids...')
        else:
            print('Norming atom ids of {}...'.format(info))
        while min(u.atoms.ids) < 1:
            u.atoms.ids += 1
    return


def norm_resids(u, info=''):
    """
    Modify existing MDA universe/atomgrp and normalize resids according to:
       - min(universe.residues.resids) > 0
       - min(universe.residues.resids) can but must not be 1

    Args:
        u (MDA universe/atomgrp): structure
        info (str): additional info for print message
            - 'reference'
            - 'topology'

    Example:
        u = mda.Universe(<top>)
        a = mda.select_atoms('protein')

        print(u.residues.resids)
        print(a.residues.resids)
        >> [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        >> [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

        norm_resids(u, 'topology')
        >> Norming topology resids...

        print(u.residues.resids)
        print(a.residues.resids)  # atomgrp is linked to universe -> same values
        >> [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
        >> [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    """
    if not isinstance(u, (mda.core.universe.Universe, mda.core.groups.AtomGroup)):
        raise TypeError('''{norm_resids.__module__}.{name_resids.__name__}():\
        \nInvalid input. u must be MDA universe/atomgrp.''')
        return

    if min(u.residues.resids) < 1:
        if info == '':
            print('Norming resids...')
        else:
            print('Norming resids of {}...'.format(info))
        while min(u.residues.resids) < 1:
            u.residues.resids += 1
    return


def norm_universe(u, info=''):
    """
    Executes the functions:
        - norm_ids(u, info)
        - norm_resids(u, info)


    For additional information, type:
        - help(<PKG_name>.norm_ids)
        - help(<PKG_name>.norm_resids)

    Args:
        u (MDA universe/atomgrp): structure
        info (str): additional info for print message
            - 'reference'
            - 'topology'
    """
    norm_ids(u, info)
    norm_resids(u, info)
    return


def norm_and_align_universe(ref, mobile):
    """
    - Norm reference and mobile universe
    - Align mobile universe on reference universe (matching atom ids and RES ids)

    For additional information, type:
        - help(<PKG_name>.norm_ids)
        - help(<PKG_name>.norm_resids)
        - help(<PKG_name>.norm_universe)
        - help(<PKG_name>.norm_and_align_universe)

    Args:
        ref    (MDA universe/atomgrp)
        mobile (MDA universe/atomgrp)
    """

    if len(ref.atoms.ids) != len(mobile.atoms.ids):
        raise ValueError(f'''{norm_and_align_universe.__module__}.{norm_and_align_universe.__name__}():\
        \nNumber of atoms doesn't match! Cannot align atom ids.''')
        return

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
        return

    if min(ref.residues.resids) < 1:
        norm_resids(ref, 'reference')

    if min(mobile.residues.resids) < 1:
        norm_resids(mobile, 'mobile')

    if np.any(ref.residues.resids != mobile.residues.resids):
        print('Reference and mobile resids are inconsistent! Aligning mobile resids to match with reference structure...')
        shift = ref.residues.resids - mobile.residues.resids
        mobile.residues.resids += shift

    print("Both universes are normed and aligned (atom ids + resids).")
    return


def true_select_atoms(u, sel='protein', ignh=True):
    """
    Create "true selection" (atomgroup copy in NEW universe) after applying
        - norm_ids()
        - norm_resids()
    Note that atom indices are reassigned according to range(0, len(u.atoms))

    Args:
        u (str): structure path or PDB id
        u (MDA universe/atomgrp): structure
        sel (str): selection string
        ignh (bool): ignore hydrogen (mass < 1.2)

    Returns:
        a (<AtomGroup>): "true selection" / atomgroup copy in NEW universe

    Additional Info:
    mda.select_atoms() remembers information about the original configuration which is sometimes NOT WANTED.

    EXAMPLE:
    # Init universe
    u = mda.Universe(<top>)

    # select_atoms() behaviour
    a = u.select_atoms("protein and type C")
    a
    >> <AtomGroup with 72 atoms>
    a.residues.atoms
    >> <AtomGroup with 964 atoms>

    # true_select_atoms() behaviour
    a = ana.true_select_atoms(u, sel="protein and type C")
    a
    >> <Universe with 72 atoms>
    a.residue.atoms
    <AtomGroup with 72 atoms>
    """
    # case 1: input is PDB ID -> fetch online
    if type(u) is str and len(u) == 4:
        u = mda.fetch_mmtf(u)
    # case 2: input is path -> create MDA Universe
    elif type(u) is str and len(u) > 4:
        u = mda.Universe(u)
    # case 3: input is MDA Universe -> just use it

    norm_universe(u)
    a0 = u.select_atoms(sel)
    if ignh:
        a0 = u.select_atoms(sel + ' and prop mass > 1.2')
    u = mda.core.universe.Merge(a0)
    a = u.atoms
    return a


def dump_structure(u, frames, save_as, default_dir="./structures", sel="protein"):
    """
    Dump structures for a list of frames with the file name "save_as".
    Automatically prepends frame number to the extension.

    Args:
        u (MDA universe/atomgrp): universe containing the structure
        frames (array/list)
        save_as (str): save name or realpath to save file
                 Note: frame number will be prepended to the extension
        default_dir (str)
        sel (str): selection string

    Returns:
        dirpath (str): realpath to directory, where structures were dumped
    """
    if "." in save_as:
        realpath = _misc.joinpath(default_dir, save_as, create_dir=True)
        dirpath = _misc.dirpath(realpath)
        base = _misc.get_base(realpath)
        ext = _misc.get_extension(realpath)

    else:
        print("specify a file format within the 'save_as' string!")
        return

    # save structures
    for ts in u.trajectory[frames]:
        frame = u.trajectory.frame
        structure = u.select_atoms(sel)
        temp_save_as = f"{dirpath}/{base}_{frame}{ext}"
        structure.write(temp_save_as)
    print("Dumped structures into:", dirpath)
    return dirpath


def shortest_RES_distances(u):
    """
    Calculates shortest RES distances (all atom level) of universe u.
    Attention: Displayed RES ids and ATM ids start with 1, not 0!

    Args:
        u (str): structure path
        u (MDA universe/atomgrp): structure

    Returns:
        SD (np.arrray): NxN Matrix with shortest RES distances
                        i.e.: SD[0][0] is shortest distance RES0-RES0
                              SD[0][1] is shortest distance RES0-RES1
                              SD[0][2] is shortest distance RES0-RES2
                              etc.
        SD_d (np.array, dtype=object): detailed List with:
            [d_min, (RES_pair), (ATM_pair), (ATM_names)]
    """
    if type(u) is str:
        u = mda.Universe(u)
    norm_universe(u)
    dim = len(u.residues.resids)
    SD = np.zeros(shape=(dim, dim))
    SD_d = []

    for i in range(dim):
        for j in range(i + 1, dim):
            DA = _distances.distance_array(u.residues[i].atoms.positions, u.residues[j].atoms.positions)
            d_min = np.min(DA)
            d_min_index = np.unravel_index(np.argmin(DA), DA.shape)  # atom indices

            # write values to list
            RES_pair = (u.residues[i].resid, u.residues[j].resid)
            ATM_pair = (u.residues[i].atoms[d_min_index[0]].id,
                        u.residues[j].atoms[d_min_index[1]].id)
            ATM_names = (u.residues[i].atoms.names[d_min_index[0]],
                         u.residues[j].atoms.names[d_min_index[1]])

            SD[i][j] = d_min
            SD_d.append([d_min, RES_pair, ATM_pair, ATM_names])

        # SD contains only half of the distances (double for loop w/o double counting RES pairs )
        SD_t = np.transpose(SD)    # -> transpose SD
        SD = np.maximum(SD, SD_t)  # -> take element-wise maximum

    # convert to np.array with dtype=object since array contains lists, tuples and sequences
    SD_d = np.array(SD_d, dtype=object)
    return(SD, SD_d)


def get_Native_Contacts(ref, d_cutoff=6.0, sel='protein', method='1'):
    """
    Calculate native contacts.

    Note:
        len(NC) < len(NC_d) because NC contains only unique RES pairs
        whereas NC_d contains each ATOM pair.

    Args:
        ref (str): reference path
        ref (MDA universe/atomgrp): reference structure
        d_cutoff (float): distance cutoff for nat. contacts
        sel (str): selection string
        method (str):
            '1' or 'Contact_Matrix': mda.contact_matrix() with d_cutoff
            '2' or 'Capped_Distance': mda.lib.distances.capped_distance() with d_cutoff #TODO
            '3' or 'Shadow_Map' #TODO

    Returns:
        NC (set): NCs (only unique RES pairs)
        NC_d (list): detailed list of NCs containing (RES pairs), (ATOM numbers), (ATOM names)
    """
    if type(method) is int:
        method = str(method)

    # load universe
    if type(ref) is str:
        u = mda.Universe(ref)
    else:
        u = ref
    norm_universe(u)
    a = u.select_atoms(sel)

    NC = set()  # set with unique NC pairs
    NC_d = []   # detailed list with NCs containing (RES pairs), (ATOM pairs), (NAME pairs)

    if method in ['1', 'Contact_Matrix']:
        RES = a.resids
        NAMES = a.names

        CM = _distances.contact_matrix(a.positions, cutoff=d_cutoff)
        for i in range(len(CM)):
            for j in range(i + 1, len(CM)):  # no double count/reduce computing time
                if RES[j] - RES[i] > 3 and CM[i][j] == True:
                    NC.add((RES[i], RES[j]))
                    NC_d.append([(RES[i], RES[j]), (i, j), (NAMES[i], NAMES[j])])

    elif method in ['2', 'Capped_Distance']:
        # TODO method 2: Capped distance
        pass
    elif method in ['3', 'Shadow_Map']:
        # TODO method 3: SHADOWMAP
        pass

    return(NC, NC_d)


def plot_Contact_Map(ref, DCA_fin=None, n_DCA=0, DCA_cols=(2, 3), DCA_skiprows="auto", filter_DCA=True, ignh=True,
                     d_cutoff=6.0, sel='protein', pdbid='pdbid', save_plot=False, **kwargs):
    """
    Create contact map based on the reference pdb structure.
    If DCA file is passed, correct/wrong contacts are visualized in green/red.

    Args:
        ref (str): reference path
        ref (MDA universe/atomgrp): reference structure
        DCA_fin (str): DCA input file (path)
        n_DCA (int): number of used DCA contacts
        DCA_cols (tuple): columns containing the RES pairs in DCA_fin
        DCA_skiprows (int): ignore header rows of DCA_fin
            -1 or "auto": auto detect
        filter_DCA (bool):
            True: ignore DCA pairs with |i-j| < 3
            False: use all DCA pairs w/o applying filter
        ignh (bool): ignore hydrogen (mass < 1.2)
        d_cutoff (float): distance cutoff for nat. contacts
        sel (str): selection string
        pdbid (str): pdbid which is used for plot title
        save_plot (bool)

    Kwargs:
        # see args of misc.figure()
    """
    # init values
    if "figsize" not in kwargs:
        kwargs = {"figsize": (7, 7)}

    # load universe
    if pdbid == 'pdbid':
        pdbid = _misc.get_PDBid(ref)
    if type(ref) is str:
        u = mda.Universe(ref)
    else:
        u = ref
    norm_universe(u)
    if ignh:
        a = true_select_atoms(u, sel, ignh=True)
    else:
        a = u.select_atoms(sel)
    res_min = min(a.resids)
    res_max = max(a.resids)

    # calculate nat. contacts for ref structure
    NC, NC_details = get_Native_Contacts(a, d_cutoff, sel)

    # PLOT
    fig, ax = _misc.figure(**kwargs)
    ax.set_aspect('equal')

    # conditions for markersize
    print("Plotting native contacts...")
    for item in NC:
        if res_max - res_min < 100:
            plt.scatter(item[0], item[1], color="silver", marker="s", s=16)
        else:
            plt.scatter(item[0], item[1], color="silver", marker="s", s=4)

    # find matching contacts ij
    if DCA_fin is not None:
        DCA = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=DCA_cols, skiprows=DCA_skiprows, filder_DCA=filter_DCA)

        if res_max - res_min < 100:
            for item in DCA:     # if item is in both lists plot it green otherwise red
                if item in NC:
                    plt.plot(item[0], item[1], color="green", marker="s", ms=4)
                else:
                    plt.plot(item[0], item[1], color="red", marker="s", ms=4)
        else:
            for item in DCA:     # if item is in both lists plot it green otherwise red
                if item in NC:
                    plt.plot(item[0], item[1], color="green", marker="s", ms=2)
                else:
                    plt.plot(item[0], item[1], color="red", marker="s", ms=2)

    # plt.legend(loc="upper left", numpoints=1)
    if res_max - res_min < 30:
        xmin, xmax = _misc.round_down(res_min, 2), _misc.round_up(res_max + 1, 2)
        ymin, ymax = _misc.round_down(res_min, 2), _misc.round_up(res_max + 1, 2)
        loc = matplotlib.ticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
    else:
        xmin, xmax = _misc.round_down(res_min, 5), _misc.round_up(res_max + 1, 5)
        ymin, ymax = _misc.round_down(res_min, 5), _misc.round_up(res_max + 1, 5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r"Residue i")
    plt.ylabel(r"Residue j")
    plt.title(f"Contact Map of {pdbid}", fontweight="bold")
    plt.tight_layout()
    if save_plot:
        _misc.savefig(filename=f"{pdbid}_Fig_Contact_Map.png", filedir="./plots")
    plt.show()
    return


def plot_DCA_TPR(ref, DCA_fin, n_DCA, DCA_cols=(2, 3), DCA_skiprows="auto", filter_DCA=True, ignh=True, d_cutoff=6.0,
                 sel='protein', pdbid='pdbid', save_plot=False, save_log=False, figsize=(8, 4.5)):
    """
    Plots true positive rate for number of used DCA contacts.

    Method:
       - calculates shortest RES distance of the selection
         (only heavy atoms if ignh is True)
       - if distance is below threshold: DCA contact is true

    Args:
        ref (str): reference path
        ref (MDA universe/atomgrp): reference structure
        DCA_fin (str): DCA input file (path)
        n_DCA (int): number of used DCA contacts
        DCA_cols (tuple/list): columns containing the RES pairs in DCA_fin
        DCA_skiprows (int): ignore header rows of DCA_fin
            -1 or "auto": auto detect
        filter_DCA (bool):
            True: ignore DCA pairs with |i-j| < 4
            False: use all DCA pairs w/o applying filter
        ignh (bool): ignore hydrogen (mass < 1.2)
        d_cutoff (float): distance cutoff for nat. contacts
        sel (str): selection string
        pdbid (str): pdbid; used for plot title and figure name
        save_plot (bool)
        save_log (bool)
        figsize (tuple)
    """
    # load universe
    if pdbid == 'pdbid':
        pdbid = _misc.get_PDBid(ref)
    if type(ref) is str:
        u = mda.Universe(ref)
    else:
        u = ref
    norm_universe(u)
    if ignh:
        a = true_select_atoms(u, sel, ignh=True)
    else:
        a = u.select_atoms(sel)
    # read DCA and calculate TPR
    DCA = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=DCA_cols, skiprows=DCA_skiprows, filter_DCA=filter_DCA)
    DCA_TPR = []  # List with TPR of DCA contacts with d < d_cutoff
    SD = shortest_RES_distances(a)[0]
    RES_min = min(a.residues.resids)
    z = 0
    for index, item in enumerate(DCA):
        if SD[item[0] - RES_min][item[1] - RES_min] <= d_cutoff:  # shift RES ids to match matrix indices
            z += 1
        percent_value = _misc.percent(z, index + 1)
        DCA_TPR.append(percent_value)

    # LOG FILE
    # List with number of DCA contacts and TPR values
    DCA_TPR_sorted = [[i + 1, DCA_TPR[i]] for i in range(len(DCA_TPR))]
    DCA_TPR_sorted = sorted(DCA_TPR_sorted, key=operator.itemgetter(1), reverse=True)  # sort by TPR value

    # PLOT: find optimum for number of DCA contacts
    # x-axis: number of DCA contacts
    # y-axis: % of DCA contacts with d <= d_cuttoff
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(range(1, n_DCA + 1), DCA_TPR, "g.", ms=8, ls="None")
        plt.plot(range(1, n_DCA + 1), DCA_TPR, "g", alpha=0.2, lw=2)
        # plt.legend(loc="upper right", numpoints=1)
        plt.xlim(0, n_DCA + 1)
        if min(DCA_TPR) < 50:
            plt.ylim(0, 105)
        else:
            plt.ylim(50, 105)
        plt.xlabel("Number of DCA Contacts")
        plt.ylabel("True Positive Rate (%)")
        plt.title(f"Protein: {pdbid}", fontweight="bold")
        sns.despine(offset=0)
        plt.tight_layout()
        if save_plot:
            _misc.savefig(filename=f"{pdbid}_Fig_DCA_TPR.png", filedir="./plots")
        if save_log:
            save_dir = _misc.mkdir("./logs")
            file_dir = save_dir + f"/{pdbid}_DCA_TPR.log"
            with open(file_dir, "w") as fout:
                fout.write("Format:\n")
                fout.write("Number of DCA Contacts \t True Positive Rate (%)\n\n")
                fout.write("10 Best values (100% excluded):\n")
                count = 0
                for i in range(len(DCA_TPR_sorted)):
                    if count < 10 and DCA_TPR_sorted[i][1] < 100.0:
                        fout.write("{}\t{}\n".format(DCA_TPR_sorted[i][0], DCA_TPR_sorted[i][1]))
                        count += 1
                fout.write("\n")
                fout.write("Full list:\n")
                fout.write("{}\t{}\n".format(0, 0.0))
                for i in range(len(DCA_TPR)):
                    fout.write("{}\t{}\n".format(i + 1, DCA_TPR[i]))
                print("Saved log as: ", file_dir)
        plt.show()
    return


def Contact_Analysis_CA(ref, top, traj, d_cutoff=6.0, sel='protein and name CA',
                        pdbid='pdbid', plot=True, save_plot=False, **kwargs):
    """
    Works currently only for CA level. All atom version is very slow.

    Args:
        ref (str): reference path
        ref (MDA universe/atomgrp): reference structure
        top (str): topology path
        top (MDA universe/atomgrp): topology
        traj (str): trajectory path
        traj (MDA universe/atomgrp): trajectory
        sel (str): selection string
        pdbid (str): pdbid; used for plot title and figure name
        plot (bool): create plot
        save_plot (bool)

    Kwargs:
        # see args of misc.figure()

    Returns:
        TIME (list): time in ps
        QValues (list): fraction of nat. contacts
        C (list): unique contact pairs IJ
        C_NperRes (list): contact number per RES (eg: RES1 has 0 contacts, RES2 has 2 contacts etc)

    """
    # load universe and shift residues if needed

    if pdbid == 'pdbid':
        pdbid = _misc.get_PDBid(ref)
    if type(ref) is str:
        ref = mda.Universe(ref)
    if type(top) is str and type(traj) is str:
        u = mda.Universe(top, traj)
    elif type(top) is str and type(traj) is not str:
        u = mda.Universe(top, traj.filename)
    elif type(top) is not str and type(traj) is str:
        u = mda.Universe(top.filename, traj)
    else:
        u = mda.Universe(top.filename, traj.filename)

    norm_universe(u)
    a = u.select_atoms(sel)
    norm_and_align_universe(ref, u)  # u is eq. to top

    # calculate nat. contacts for ref structure
    NC, NC_details = get_Native_Contacts(ref, d_cutoff, sel)

    # calculate contacts for trajectory
    TIME = []  # time
    QValues = []
    C = []  # unique contact pairs (RES/IJ pairs)
    C_NperRES = []  # contact number per RES (eg: RES1 has 0 contacts, RES2 has 2 contacts etc)

    print("Performing contact analysis for trajectory.")
    for ts in tqdm(u.trajectory):
        TIME.append(u.trajectory.time)
        dist_array = _distances.distance_array(a.positions, a.positions)

        ignore_asr = 3  # asr: adjecent sequence residues
        for i in range(ignore_asr + 1):  # left half of main diag
            np.fill_diagonal(dist_array[i:], 999999999)
        dist_array = np.matrix.transpose(dist_array)  # right half of main diag
        for i in range(ignore_asr + 1):
            np.fill_diagonal(dist_array[i:], 999999999)
        dist_array = np.matrix.transpose(dist_array)

        C_NperRES_now = np.sum(d <= d_cutoff for d in dist_array)
        C_NperRES.append(C_NperRES_now)

        # get unique contact pairs
        IJ = np.where(dist_array <= d_cutoff)
        IJ = np.transpose(IJ)
        IJ = [(item[0] + 1, item[1] + 1) for item in IJ if item[0] < item[1]]  # index shift for resids: 0..n-1 --> 1..n
        C.append(IJ)

        # get QValues
        num = 0.0
        div = len(NC)
        for item in IJ:
            if item in NC:
                num += 1
        QValues.append(round(num / div, 2))

    # PLOT
    if plot:
        TIME_ns = [item * 0.001 for item in TIME]
        C_perFrame = [len(item) for item in C]

        fig, ax = _misc.figure(**kwargs)
        plt.plot(TIME_ns, QValues, "r.", ms=1)
        plt.plot(TIME_ns, QValues, "r", lw=1.5, alpha=0.3)
        # ax.set_ylim(_misc.round_down(min(QValues), 0.1), _misc.round_up(max(QValues), 0.1))
        ax.set(xlabel="Time (ns)", ylabel="Q")
        plt.title("QValues, avg = {:.2f}".format(np.mean(QValues)), fontweight="bold")
        plt.tight_layout()
        if save_plot:
            _misc.savefig(filename=f"{pdbid}_Fig_QValues.png", filedir="./plots")
        plt.show()

        fig, ax = _misc.figure()
        plt.plot(TIME_ns, C_perFrame, "r.", ms=1)
        plt.plot(TIME_ns, C_perFrame, "r", lw=1.5, alpha=0.3)
        ax.set_ylim(_misc.round_down(min(C_perFrame), 5), _misc.round_up(max(C_perFrame), 5))
        ax.set(xlabel="Time (ns)", ylabel="Contacts")
        plt.title("Contacts, avg = {:.2f}".format(np.mean(C_perFrame)), fontweight="bold")
        plt.tight_layout()
        if save_plot:
            _misc.savefig(filename=f"{pdbid}_Fig_Contacts.png", filedir="./plots")
        plt.show()

    return(TIME, QValues, C, C_NperRES)


################################################################################
################################################################################
### analysis.py plot functions


def get_trendline(xdata, ydata, compress=10):
    """
    Reduces <xdata, ydata> by <compress> factor and returns trendline data.
    (Trendline: sum and normalize each <compress> elements of input data)

    Args:
        xdata (list/array)
        ydata (list/array)
        compress (int): compress factor (sum and normalize <compress> elements)

    Returns:
        trend_xdata (list/array)
        trend_ydata (list/array)
    """
    if len(xdata) != len(ydata):
        raise ValueError("xdata and ydata have unequal length.")

    trend_xdata = np.add.reduceat(xdata, np.arange(0, len(xdata), compress))/float(compress)
    trend_ydata = np.add.reduceat(ydata, np.arange(0, len(ydata), compress))/float(compress)
    trend_xdata = trend_xdata.tolist()
    trend_ydata = trend_ydata.tolist()

    # fix normalization of last element if data is not fully divisible by compress (ie remainder exists)
    remainder = len(xdata) % compress
    if remainder != 0:
        trend_xdata = trend_xdata[: -1] + [sum(xdata[-remainder:])/float(remainder)]
        trend_ydata = trend_ydata[: -1] + [sum(ydata[-remainder:])/float(remainder)]

    if isinstance(xdata, np.ndarray):
        trend_xdata = np.array(trend_xdata)
    if isinstance(ydata, np.ndarray):
        trend_ydata = np.array(trend_ydata)
    return trend_xdata, trend_ydata


def plot_trendline(xdata, ydata, compress=10, fig=None, **kwargs):
    """
    Plot trendline of <xdata, ydata> and return trendline object
    Remove trendline via trendline.remove()

    Args:
        xdata (list/array)
        ydata (list/array)
        compress (int): compress factor (sum and normalize <compress> elements)
        fignum (int): figure number (get with plt.gcf().number)

    Kwargs:
        "alpha": 1.0
        "color": "black"
        "ls": "-"
        "lw": 2.0
        "ms": 1
        "marker": "

    Returns:
        trendline (matplotlib.lines.Line2D): trendline object
    """
    default = {"alpha": 1.0,
               "color": "black",
               "label": None,
               "ls": "-",
               "lw": 2.0,
               "ms": 1,
               "marker": "."}
    cfg = _misc.CONFIG(default, **kwargs)
    #####################################
    # get trendline data
    trend_xdata, trend_ydata = get_trendline(xdata, ydata, compress=compress)

    # use existing figure
    if isinstance(fig, int):
        fig = plt.figure(num=fig)
    elif isinstance(fig, matplotlib.figure.Figure):
        fig = plt.figure(num=fig.number)
    else:
        fig = plt.gcf()

    # remove existing trendline
    try:
        fig.trendline.remove()
        del fig.trendline
    except AttributeError:
        pass  # no trendline found -> do nothing

    # plot new trendline
    trendline = plt.plot(trend_xdata, trend_ydata,
                         alpha=cfg.alpha, color=cfg.color, label=cfg.label,
                         ls=cfg.ls, lw=cfg.lw, ms=cfg.ms, marker=cfg.marker)

    fig.trendline = trendline[0]
    return fig.trendline


def remove_trendline(trendline=None, fig=None):
    """
    """
    if trendline is not None:
        trendline.remove()
        del trendline
        return
    elif fig is not None:
        fig.trendline.remove()
        del fig.trendline
        return
    elif trendline is None and fig is None:
        fig = plt.gcf()
    try:
        fig.trendline.remove()
        del fig.trendline
    except AttributeError:
        print("Figure has no trendline object.")
    return


def PLOT(xdata, ydata, xlabel='', ylabel='', title='', xlim=None, ylim=None, **kwargs):
    """
    General plot function for analysis.py

    Args:
        xdata (array/list)
        ydata (array/list)
        xlabel (str)
        ylabel (str)
        title (None/str)
        xlim (None/list)
        ylim (None/list)

    Kwargs:
        # see args of misc.figure()

        "alpha": 0.3
        "color": "r"
        "lw": 1.5
        "ms": 1
        "marker": "."

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)

    Example:
        >> PLOT(TIME, RMSD, xlabel="time (ns)", ylabel=r"RMSD ($\AA$)", title="RMSD plot",
                xlim=[50, 500], ylim=[0, 20])
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"alpha": 0.3,
               "color": "r",
               "lw": 1.5,
               "ms": 1,
               "marker": "."}
    cfg = _misc.CONFIG(default, **kwargs)
    #####################################
    fig, ax = _misc.figure(**kwargs)
    cp = sns.color_palette(None)

    _dim_xdata = len(np.shape(xdata))
    if _dim_xdata == 1:
        plt.plot(xdata, ydata, color=cfg.color, ls="", marker=cfg.marker, ms=cfg.ms)
        plt.plot(xdata, ydata, color=cfg.color, lw=cfg.lw, alpha=cfg.alpha)
    else:
        for i in range(len(xdata)):
            j = i % 10  # sns color_palette has usually 10 colors
            plt.plot(xdata[i], ydata[i], color=cp[j], ls="", marker=cfg.marker, ms=cfg.ms)
            plt.plot(xdata[i], ydata[i], color=cp[j], lw=cfg.lw, alpha=cfg.alpha)

    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    if title != "" or title != None:
        plt.title(title, fontweight='bold')

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.show()

    return(fig, ax)


def plot_RMSD(RMSD_file, sss=[None, None, 10], verbose=None, save_as="", **kwargs):
    """
    Args:
        RMSD_file (str): path to RMSD file
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        verbose (None/bool):
            None: print step to user (as a reminder)
            True: print start, step, stop to user
            False: no print
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        alpha (float)
        color (str)
        cut_min (None/int/float): min value for cutoff
        cut_max (None/int/float): max value for cutoff
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        filedir (str): default directory where to save figure
        title (None/str)

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
    """
    #######################################
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"alpha": 0.3,
               "color": "r",
               "lw": 1.5,
               "ms": 1,
               "marker": ".",
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "cut_min": None,
               "cut_max": None,
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)

    if verbose == None:
        print("Plottinig data with stepsize of", cfg.step)
    elif verbose == True:
        print("Plotting data with parameter values:")
        print("\tstart:", cfg.start)
        print("\tstop:", cfg.stop)
        print("\tstep:", cfg.step)
    header = _misc.autodetect_header(RMSD_file)

    TIME = np.loadtxt(RMSD_file, skiprows=header, usecols=(0,))
    TIME = [round(x, 3) for x in TIME[cfg.start:cfg.stop:cfg.step]]
    RMSD = np.loadtxt(RMSD_file, skiprows=header, usecols=(1,))
    RMSD = RMSD[cfg.start: cfg.stop: cfg.step]

    # apply cutoff and force same dimensions by tracking indices
    if cfg.cut_min is not None or cfg.cut_max is not None:
        RMSD, RMSD_ndx = _misc.get_cutoff_array(RMSD, cfg.cut_min, cfg.cut_max)
        TIME = _misc.get_sub_array(TIME, RMSD_ndx)

    # plot
    fig, ax = _misc.figure(**cfg)
    plt.plot(TIME, RMSD,
             color=cfg.color, ls="", marker=cfg.marker, ms=cfg.ms)
    plt.plot(TIME, RMSD,
             color=cfg.color, lw=cfg.lw, alpha=cfg.alpha)

    if cfg.title is not None:
        plt.title(cfg.title, fontweight='bold')

    plt.tight_layout()
    plt.show()

    if save_as != "":
        if "filedir" in kwargs:
            _misc.savefig(filename=save_as, filedir=kwargs["filedir"])
        else:
            _misc.savefig(filename=save_as)
        plt.savefig(save_as, dpi=300)

    return(fig, ax)


def _HELP_setup_bins(data, vmin=None, vmax=None, vbase=1, bins=None, n_bins=200, n_bins_autoadd=1):
    """
    HELP function to setup bins for histogram. Returns a dictionary to update
    important cfg parameter, such as vmin, vmax.

    Args:
        data (array/list)
        vmin (None/int/float): min value of histogram/bins
        vmax (None/int/float): max value of histogram/bins
        vbase (int): rounding base of vmin, vmax.
                     see misc.round_down(number,base) or misc.round_up(number,base)
        bins (None/list): use bins if passed, else construct bins based on n_bins, vmin, vmax, vbase
        n_bins (int): number of bins
        n_bins_autoadd (int): autocorrect n_bins (default: 1), because hist usually require 1 extra bin

    Returns
        bins (list)
        bins_step (int/float)
        cfg (dict): dictionary containing the current values of:
            vmin, vmax, vbase, bins, n_bins, n_bins_autoadd
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = dict(vmin=vmin, vmax=vmax, vbase=vbase,
                   bins=bins, n_bins=n_bins, n_bins_autoadd=n_bins_autoadd)
    cfg = _misc.CONFIG(default)

    if cfg.vmin is None:
        min_ = min(_misc.flatten_array(data))
        cfg.vmin = _misc.round_down(min_, cfg.vbase)
    if cfg.vmax is None:
        max_ = max(_misc.flatten_array(data))
        cfg.vmax = _misc.round_up(max_, cfg.vbase)
    if cfg.bins is None:
        cfg.bins = np.linspace(cfg.vmin, cfg.vmax, cfg.n_bins+cfg.n_bins_autoadd)
        bins = cfg.bins
        bins_step = cfg.bins[1]-cfg.bins[0]
    else:
        bins = cfg.bins
        bins_step = cfg.bins[1]-cfg.bins[0]
        cfg.n_bins = len(bins)-1
    return(bins, bins_step, cfg)


def plot_hist(data, sss=[None, None, None], save_as="", **kwargs):
    """
    Args:
        data (list/array/list of lists): input data. Dimensions of each list must
                                         be equal if multiple lists are provided.
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        cut_min (None/int/float): min value for cutoff
        cut_max (None/int/float): max value for cutoff
        apply_cut_limits (bool): apply plt.xlim(cut_min, cut_max) if orientation is "vertical"
                                 apply plt.ylim(cut_min, cut_max) if orientation is "horizontal"
        align_bins (str): 'center', 'edge' (default, left edge alignment)
        bins (None/list): use bins if passed, else construct bins based on n_bins, vmin, vmax, vbase
        n_bins (int): number of bins
        n_bins_autoadd (int): autocorrect n_bins (default: 1), because code requires 1 extra bin.
        vmin (None/int/float): min value of histogram/bins
        vmax (None/int/float): max value of histogram/bins
        vbase (int): rounding base of vmin, vmax.
                     see misc.round_down(number,base) or misc.round_up(number,base)
        logscale (bool): apply logscale on the "count" axis
        minorticks (bool): turns minorticks (~logscale) for hist figure on/off
        norm (bool): normalize histogram
        orientation (str): "vertical", "horizontal"
        alpha (int/float)
        colors (list of str/sns.colorpalette)
        ec (None/str): edge color
        title (None/str)

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)

        hist (tuple): n, bins, patches
        or HIST (list): list of hist, i.e. list of (n, bins, patches) tuples
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "cut_min": None,
               "cut_max": None,
               "apply_cut_limits": False,

               "align_bins": "edge",
               "bins": None,
               "n_bins": 100,
               "n_bins_autoadd": 1,
               "vmin": None,
               "vmax": None,
               "vbase": 1,

               "orientation": "vertical",
               "norm": False,
               "logscale": False,
               "minorticks": False,

               "alpha": 0.5,
               "colors": ["red", "grey"],
               "ec": "white",
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)
    # edge color has to be a string, so None will result into white edge color; fix:
    if cfg.ec is None:
        cfg.ec = "None"
    ##############################################################
    ### detect if data is single or multiple arrays
    # single array
    if len(np.shape(data)) == 1:
        if isinstance(data, (type([]), np.ndarray)):
            data_length = 1

    # multiple arrays
    else:
        if isinstance(data[0], (type([]), np.ndarray)):
            data_length = len(data)
        elif isinstance(data[0], type(str)):
            data_length = len(data)
        if data_length > 2:
            cfg.colors = sns.color_palette(None, data_length)

    # apply cutoff if specified
    if data_length == 1:
        data, _ = _misc.get_cutoff_array(data[cfg.start: cfg.stop: cfg.step],
                                         cfg.cut_min, cfg.cut_max)
    else:
        data_temp = list(data)
        data = []
        for item in data_temp:
            cutoff_data, _ = _misc.get_cutoff_array(item[cfg.start: cfg.stop: cfg.step],
                                                    cfg.cut_min, cfg.cut_max)
            data.append(cutoff_data)

    ##############################################################
    ########################  histogram  #########################
    ##############################################################
    # setup bins
    bins, bins_step, sub_cfg = _HELP_setup_bins(data=data, vmin=cfg.vmin, vmax=cfg.vmax, vbase=cfg.vbase,
                                                bins=cfg.bins, n_bins=cfg.n_bins, n_bins_autoadd=cfg.n_bins_autoadd)
    cfg.update_config(**sub_cfg)

    if "ax" in cfg.keys():
        plt.sca(cfg["ax"])
    else:
        fig, ax = _misc.figure(**cfg)
    # plt.hist version
    if data_length == 1:
        hist = plt.hist(data, bins=bins,
                        density=cfg.norm, log=cfg.logscale,
                        orientation=cfg.orientation, stacked=True,
                        alpha=cfg.alpha, ec=cfg.ec, histtype="barstacked")

    else:
        HIST = []
        for ndx, item in enumerate(data):
            hist = plt.hist(item, bins=bins,
                            density=cfg.norm, log=cfg.logscale,
                            orientation=cfg.orientation, stacked=True,
                            alpha=cfg.alpha, ec=cfg.ec, histtype="barstacked")
            HIST.append(hist)

    # plt.bar/plt.barh version (redraw -> needed for pickle functions)
    if "num" in cfg.keys():
        del cfg["num"]
    fig, ax = _misc.figure(num=fig.number, **cfg)
    if data_length == 1:
        if cfg.orientation == "vertical":
            h = hist[0]
            w = bins_step
            plt.bar(bins[: -1], height=h, width=w, align=cfg.align_bins,
                    color=cfg.colors[0], alpha=cfg.alpha, ec=cfg.ec, log=cfg.logscale)
        if cfg.orientation == "horizontal":
            h = bins_step
            w = hist[0]
            plt.barh(bins[: -1], height=h, width=w, align=cfg.align_bins,
                     color=cfg.colors[0], alpha=cfg.alpha, ec=cfg.ec, log=cfg.logscale)

    else:
        for ndx, item in enumerate(data):
            if cfg.orientation == "vertical":
                h = HIST[ndx][0]
                w = bins_step
                plt.bar(bins[: -1], height=h, width=w, align=cfg.align_bins,
                        color=cfg.colors[ndx], alpha=cfg.alpha, ec=cfg.ec, log=cfg.logscale)
            if cfg.orientation == "horizontal":
                h = bins_step
                w = HIST[ndx][0]
                plt.barh(bins[: -1], height=h, width=w, align=cfg.align_bins,
                         color=cfg.colors[ndx], alpha=cfg.alpha, ec=cfg.ec, log=cfg.logscale)

    if cfg.orientation == "vertical":
        if cfg.apply_cut_limits:
            plt.xlim([cfg.cut_min, cfg.cut_max])
        else:
            plt.xlim([cfg.vmin, cfg.vmax])
    elif cfg.orientation == "horizontal":
        if cfg.apply_cut_limits:
            plt.ylim([cfg.cut_min, cfg.cut_max])
        else:
            plt.ylim([cfg.vmin, cfg.vmax])
    if cfg.minorticks == False:
        plt.minorticks_off()

    if cfg.title is not None:
        plt.title(cfg.title, fontweight='bold')
    if save_as != "":
        _misc.savefig(save_as)

    plt.tight_layout()

    if data_length == 1:
        return(fig, ax, hist)
    else:
        return(fig, ax, HIST)


def plot_deltahist(RMSD_file, RMSD_ref, sss=[None, None, None],
                   show_all_hist=False, save_as="", **kwargs):
    """
    Args:
        RMSD_file (str): path to RMSD file
        RMSD_ref (str): path to RMSD reference file
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        show_all_hist (bool):
            True:  show ref hist, non-ref hist and delta hist
            False: show only delta hist
        title (str)
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        alpha (float)
        colors (list): two string elements in list: color_positive, color_negative
                       Example: ["g", "r"]
        color_positive (str)
        color_negative (str)
        ec (None/str): edge color
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        cut_min (None/int/float): min value for cutoff (here: alias of vmin)
        cut_max (None/int/float): max value for cutoff (here: alias of vmax)
        apply_cut_limits (bool): apply plt.xlim(cut_min, cut_max) if orientation is "vertical"
                                 apply plt.ylim(cut_min, cut_max) if orientation is "horizontal"
        bins (None/list): use bins if passed, else construct bins based on n_bins, vmin, vmax, vbase
        n_bins (int): number of bins
        n_bins_autoadd (int): autocorrect n_bins (default: 1), because code requires 1 extra bin.
        vmin (None/int/float): min value of histogram/bins
        vmax (None/int/float): max value of histogram/bins
        vbase (int): rounding base of vmin, vmax.
                     see misc.round_down(number,base) or misc.round_up(number,base)
        logscale (bool): apply logscale on the "count" axis
        minorticks (bool): turns minorticks (~logscale) for hist figure on/off
        norm (bool): normalize histogram
        orientation (str): "vertical", "horizontal"
        title (None/str)

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "cut_min": None,
               "cut_max": None,
               "apply_cut_limits": False,

               "bins": None,
               "n_bins": 200,
               "n_bins_autoadd": 1,
               "vmin": None,
               "vmax": None,
               "vbase": 0.2,

               "logscale": True,
               "minorticks": False,
               "norm": False,
               "orientation": "horizontal",

               "alpha": 0.8,
               "colors": ["g", "r"],
               "ec": "None",
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)
    cfg.update_by_alias(alias="color_positive", key="colors", key_ndx=0, **kwargs)
    cfg.update_by_alias(alias="color_negative", key="colors", key_ndx=1, **kwargs)
    cfg.update_by_alias(alias="cut_min", key="vmin", **kwargs)
    cfg.update_by_alias(alias="cut_max", key="vmax", **kwargs)

    header = _misc.autodetect_header(RMSD_ref)
    RMSD = np.loadtxt(RMSD_file, skiprows=header, usecols=(1,))
    RMSD_ref = np.loadtxt(RMSD_ref, skiprows=header, usecols=(1,))

    if [cfg.start, cfg.stop, cfg.step] != [0, -1, 1]:
        RMSD = RMSD[cfg.start: cfg.stop: cfg.step]
        RMSD_ref = RMSD_ref[cfg.start: cfg.stop: cfg.step]
    ##############################################################
    ################# ref and non-ref histogram ##################
    ##############################################################
    # setup bins
    bins, bins_step, sub_cfg = _HELP_setup_bins(data=RMSD_ref, vmin=cfg.vmin, vmax=cfg.vmax, vbase=cfg.vbase,
                                                bins=cfg.bins, n_bins=cfg.n_bins, n_bins_autoadd=cfg.n_bins_autoadd)
    cfg.update_config(**sub_cfg)

    fig, ax = _misc.figure(grid=[2, 1], **cfg)
    plt.sca(ax[0])
    plt.title("ref histogram")
    hist_ref = plt.hist(RMSD_ref, bins,
                        density=cfg.norm, log=cfg.logscale, orientation=cfg.orientation,
                        alpha=cfg.alpha, ec=cfg.ec, histtype="barstacked", stacked=True)
    plt.sca(ax[1])
    plt.title("non-ref histogram")
    hist = plt.hist(RMSD, bins,
                    density=cfg.norm, log=cfg.logscale, orientation=cfg.orientation,
                    alpha=cfg.alpha, ec=cfg.ec, histtype="barstacked", stacked=True)
    plt.tight_layout()
    ##############################################################
    ###################### delta histogram #######################
    ##############################################################
    diff = (hist[0]-hist_ref[0])  # difference of ref and non-ref hist
    positive = [i if i > 0 else 0 for i in diff]  # positive subset including zeros
    negative = [i if i < 0 else 0 for i in diff]  # negative subset including zeros
    if cfg.logscale:
        abs_negative = [abs(i) for i in negative]   # absolute negative subset including zeros

    if show_all_hist:
        fig, ax = _misc.figure(**kwargs)
    else:
        if "num" not in kwargs:
            fig, ax = _misc.figure(num=fig.number, **kwargs)  # reuse canvas by passing fig number
        else:
            fig, ax = _misc.figure(**kwargs)

    if cfg.orientation == "horizontal":
        if cfg.logscale:
            # positive delta ~ green; fix dim by ignoring last element
            plt.barh(bins[: -1], width=positive, height=bins_step,
                     color=cfg.colors[0], ec=cfg.ec, alpha=cfg.alpha, log=True)
            # negative delta ~ red; fix dim by ignoring last element
            plt.barh(bins[: -1], width=abs_negative, height=bins_step,
                     color=cfg.colors[1], ec=cfg.ec, alpha=cfg.alpha, log=True)
        else:
            # positive delta ~ green; fix dim by ignoring last element
            plt.barh(bins[: -1], width=positive, height=bins_step,
                     color=cfg.colors[0], ec='None', alpha=cfg.alpha, log=False)
            # negative delta ~ red; fix dim by ignoring last element
            plt.barh(bins[: -1], width=negative, height=bins_step,
                     color=cfg.colors[1], ec='None', alpha=cfg.alpha, log=False)
        if cfg.apply_cut_limits:
            plt.ylim(cfg.vmin, cfg.vmax)  # here: vmin/vmax alias of cut_min/cut_max
    elif cfg.orientation == "vertical":
        if cfg.logscale:
            # positive delta ~ green; fix dim by ignoring last element
            plt.bar(bins[: -1], height=positive, width=bins_step,
                    color=cfg.colors[0], ec=cfg.ec, alpha=cfg.alpha, log=True)
            # negative delta ~ red; fix dim by ignoring last element
            plt.bar(bins[: -1], height=abs_negative, width=bins_step,
                    color=cfg.colors[1], ec=cfg.ec, alpha=cfg.alpha, log=True)
        else:
            # positive delta ~ green; fix dim by ignoring last element
            plt.bar(bins[: -1], height=positive, width=bins_step,
                    color=cfg.colors[0], ec=cfg.ec, alpha=cfg.alpha, log=False)
            # negative delta ~ red; fix dim by ignoring last element
            plt.bar(bins[: -1], height=negative, width=bins_step,
                    color=cfg.colors[1], ec=cfg.ec, alpha=cfg.alpha, log=False)
        if cfg.apply_cut_limits:
            plt.xlim(cfg.vmin, cfg.vmax)  # here: vmin/vmax alias of cut_min/cut_max

    if cfg.minorticks == False:
        plt.minorticks_off()

    if cfg.title != None:
        plt.title(cfg.title, fontweight='bold')
    if save_as != "":
        _misc.savefig(save_as)
    plt.tight_layout()
    return(fig, ax)


def _HELP_convert_RMSD_nm2angstrom(RMSD_nm):
    """
    Helpfunction for plot_HEATMAP_REX_RMSD():
    convert RMSD values: RMSD_nm -> RMSD_anstrom

    Args:
        RMSD_nm (list): rmsd values in nm

    Returns:
        RMSD (list): rmsd values in angstrom
    """
    RMSD_max = np.amax(RMSD_nm)
    if RMSD_max < 5.0:
        RMSD = [x*10 for x in RMSD_nm]
    return RMSD


def _HELP_convert_xticks_frame2time(ax, delta_t=0.002):
    """
    Helpfunction for plot_HEATMAP_REX_RMSD():
    convert ax xticks: frame -> time (ns)

    Args:
        ax (matplotlib.axes._subplots.Axes)

    Returns:
        xticks_time (list)
    """
    xticks_time = []
    for item in ax.get_xticklabels():
        xticks_time.append(item.get_text())
    xticks_time = [float(x)*delta_t for x in xticks_time]
    xticks_time = [int(x) if x.is_integer() else x for x in xticks_time]
    return xticks_time


def plot_HEATMAP(data, save_as="", **kwargs):
    """
    Note: if data is incomplete:
       -> use min/max values or np.nan for missing values
          and the keyword argument annot=True or annot=False

    Args:
        data (arrays)
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        annot (bool/list): toggle annotation within the heatmap squares.
                           if a list is provided, annotates the list content.
        cmap (str): color map string
        n_colors (None/int)
        cbar_min / vmin (None/int/float): min value of colorbar and heatmap
        cbar_max / vmax (None/int/float): max value of colorbar and heatmap
        cbar_label (None/str)
        title (None/str)
        xlabel (None/str)
        ylabel (None/str)
        xticklabels (None/list)
        yticklabels (None/list)
        show_ticks (bool): toggle (label) ticks

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax ~ matplotlib.axes._subplots.Axes)
    """
    default = {"figsize": (6.5, 4),
               "vmin": None,
               "vmax": None,

               "annot": True,
               "cmap": "RdBu_r",
               "n_colors": 50,
               "cbar_min": None,
               "cbar_max": None,
               "cbar_label": None,

               "title": None,
               "xlabel": None,
               "ylabel": None,
               "xticklabels": None,
               "yticklabels": None,
               "show_ticks": False}
    cfg = _misc.CONFIG(default, **kwargs)
    cfg.update_by_alias(alias="cbar_min", key="vmin", **kwargs)
    cfg.update_by_alias(alias="cbar_max", key="vmax", **kwargs)

    if cfg.xticklabels is None:
        ydim, xdim = np.shape(data)
        cfg.xticklabels = range(xdim)
    if cfg.yticklabels is None:
        ydim, xdim = np.shape(data)
        cfg.yticklabels = range(ydim)

    fig, ax = _misc.figure(**cfg)
    # settings
    cmap = sns.color_palette(cfg.cmap, cfg.n_colors)  # Red Blue reversed, 50 colors
    hm = sns.heatmap(data, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax,
                     linewidth=0, annot=cfg.annot,
                     xticklabels=cfg.xticklabels, yticklabels=cfg.yticklabels)

    if cfg.xlabel is not None:
        plt.xlabel(cfg.xlabel, fontweight='bold')
    if cfg.ylabel is not None:
        plt.ylabel(cfg.ylabel, fontweight='bold')
    if cfg.show_ticks == False:
        ax.tick_params(left=False, bottom=False)
        locs = ax.get_yticks()
        ax.set_yticks(locs-0.1)
    #####################################################
    # set cbar label
    cbar = hm.collections[0].colorbar
    if cfg.cbar_label is not None:
        cbar.set_label(cfg.cbar_label, fontweight='bold')
    #####################################################
    if cfg.title is not None:
        plt.title(cfg.title, fontweight='bold')

    plt.tight_layout()

    if save_as != "":
        if ".pdf" in save_as:
            print("""Too much data within the heatmap to save it as a .pdf file.
    (vector graphic ~ number of objects too high)
    Saving figure as .png with 300 dpi instead...""")
            save_as = f"{_misc.dirpath(save_as)}/{_misc.get_base(save_as)}.png"
        _misc.savefig(filename=save_as)

    return(fig, ax)


def plot_HEATMAP_REX_RMSD(REX_RMSD_dir, cps=["RdBu_r", 50, 0, 8], auto_convert=True, save_as="", **kwargs):
    """
    Args:
        REX_RMSD_dir (str): directory to REX_RMSD files
        cps (list): color_palette_settings
            [0]: color_palette name (str)
            [1]: n_colors (int)
            [2]: heatmap vmin (int/float)
            [3]: heatmap vmax (int/float)
        auto_convert (bool): convert "displayed" RMSD and xticks for plot
                             returned values stay unconverted
            if True:
                RMSD: RMSD (nm) -> RMSD (angstrom)
                xticks: frame -> time (ns)
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        title (None/str)

    Returns:
        TIME (list): time values of first REX_RMSD file
        RMSD (list of arrays): RMSD values of each REX_RMSD file
    """
    ### setup default kwargs if not passed
    default = {"figsize": (7.5, 6),
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)

    RMSD_FILES = []
    for file in sorted(glob.glob(os.path.join(REX_RMSD_dir, '*.xvg'))):
        RMSD_FILES.append(file)
    TIME = np.loadtxt(RMSD_FILES[0], skiprows=_misc.autodetect_header(file), usecols=(0,))
    TIME = [round(x, 3) for x in TIME]
    RMSD = [np.loadtxt(file, skiprows=_misc.autodetect_header(file), usecols=(1,)) for file in RMSD_FILES]

    fig, ax = _misc.figure(**cfg)
    # settings
    cmap = sns.color_palette(cps[0], cps[1])  # Red Blue reversed, 50 colors
    vmin = cps[2]
    vmax = cps[3]
    xticklabels = 25000  # show every 25k label
    yticklabels = 10     # show every 10 label

    if auto_convert:
        RMSD_temp = _HELP_convert_RMSD_nm2angstrom(RMSD)
        RMSD_max = np.amax(RMSD_temp)
        hm = sns.heatmap(RMSD_temp, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         cbar_kws=dict(ticks=np.arange(vmin, vmax+1, 2)))  # set custom cbar ticks
        ax.set_xticklabels(_HELP_convert_xticks_frame2time(ax))  # convert time ticks
        plt.xlabel('Time (ns)', fontweight='bold')
    else:
        RMSD_max = np.amax(RMSD)
        hm = sns.heatmap(RMSD, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         cbar_kws=dict(ticks=np.arange(vmin, vmax+1, 2)))  # set custom cbar ticks
        plt.xlabel('Frame Number', fontweight='bold')

    plt.ylabel('Replica Number', fontweight='bold')
    ax.invert_yaxis()  # (sns.heatmap() plots array values from top to bottom) -> reverse axis
    #####################################################
    # set cbar label
    cbar = hm.collections[0].colorbar
    if RMSD_max < 5.0:
        cbar.set_label('RMSD (nm)', fontweight='bold')
    else:
        cbar.set_label('RMSD ($\AA$)', fontweight='bold')
    #####################################################
    if cfg.title is not None:
        plt.title(cfg.title, fontweight='bold')

    plt.tight_layout()
    plt.tight_layout()
    plt.show()

    if save_as != "":
        if ".pdf" in save_as:
            print("""Too much data within the heatmap to save it as a .pdf file.
    (vector graphic ~ number of objects too high)
    Saving figure as .png with 300 dpi instead...""")
            save_as = f"{_misc.get_base(save_as)}.png"
        _misc.savefig(filename=save_as)

    return(TIME, RMSD)

################################################################################
################################################################################
# GDT: Global Distance Test


def get_array_percent(dist_array, cutoff):
    """
    Get percentage and indices of dist_array that fulfil the condition:
        dist_array <= cutoff

    Args:
        dist_array (array): distance array
        cutoff (int/float)

    Returns:
        p (float): percent of dist_array (condition: dist_array <= cutoff)
        ndx (array): indices of dist_array (condition:
        dist_array <= cutoff)
    """
    norm = len(dist_array)
    ndx = np.where(dist_array <= cutoff)
    p = _misc.percent(len(ndx[0]), norm)
    return(p, ndx)


def get_PairDistances(mobile, ref, sel="protein and name CA", mobile_sel="auto", ref_sel="auto", weights="mass"):
    """
    Aligns mobile to ref and calculates PairDistances (e.g. CA-CA distances).

    Args:
        mobile (MDA universe/atomgrp): mobile structure with trajectory
        ref (MDA universe/atomgrp): reference structure
        sel (str): selection string
        mobile_sel (str/MDA atomgrp)
        ref_sel (str/MDA atomgrp)
        weights

    Returns:
        PAIR_DISTANCES
        _resids_mobile (array)
        _resids_ref (array)
        _RMSD (tuple): (RMSD before alignment, RMSD after alignment)
        """
    # Test if input is either class or string (otherwise runtime error due to == operator)
    if isinstance(mobile_sel, mda.core.groups.AtomGroup):
        pass
    elif mobile_sel == "auto":
        mobile_sel = mobile.select_atoms(sel)
    else:
        raise TypeError(f'''{get_PairDistances.__module__}.{get_PairDistances.__name__}(): mobile_sel has to be either:\
        \n    class: <MDAnalysis.core.groups.AtomGroup>\
        \n    str: 'auto'.''')

    if isinstance(ref_sel, mda.core.groups.AtomGroup):
        pass
    elif ref_sel == "auto":
        ref_sel = ref.select_atoms(sel)
    else:
        raise TypeError(f'''{get_PairDistances.__module__}.{get_PairDistances.__name__}(): ref_sel has to be either:\
        \n    class: <MDAnalysis.core.groups.AtomGroup>\
        \n    str: 'auto'.''')

    # get_PairDistances function
    RMSD = _align.alignto(mobile, ref, select=sel, weights=weights)
    _resids_mobile, _resids_ref, PAIR_DISTANCES = mda.analysis.distances.dist(mobile_sel, ref_sel)
    return(PAIR_DISTANCES, RMSD, _resids_mobile, _resids_ref)


def _HELP_sss_None2int(obj, cfg):
    """
    Note: this function is (probably) required when a custom function uses
    either a MDA universe or (distance)matrix and a misc.CONFIG class with
    the keys "sss", "start", "stop", "step".

    Reason:
        slicing: aList[None:None:None] is equal to aList[::]
        get item: aList[None] does not exist

    Converts None entries of the config keys to integers:
        sss = [None, None, None] -> [0, max frame of MDA universe, 1]
        start = None ->  0
        stop = None ->  max frame of u
        step = None ->  1

    Args:
        obj (MDA universe / matrix): MDA universe or distane(matrix)
        cfg (misc.CONFIG class)

    Returns:
        cfg (misc.CONFIG class)
    """
    if isinstance(obj, mda.core.universe.Universe):
        if "sss" in cfg.keys():
            if cfg.sss[0] == None:
                cfg.sss[0] = 0
            if cfg.sss[1] == None:
                cfg.sss[1] = obj.trajectory.n_frames
            if cfg.sss[2] == None:
                cfg.sss[2] = 1

        if "start" in cfg.keys() or "stop" in cfg.keys() or "step" in cfg.keys():
            if cfg.start == None:
                cfg.start = 0
            if cfg.stop == None:
                cfg.stop = obj.trajectory.n_frames
            if cfg.step == None:
                cfg.step = 1

    elif isinstance(obj, (np.ndarray, list)):
        if "sss" in cfg.keys():
            if cfg.sss[0] == None:
                cfg.sss[0] = 0
            if cfg.sss[1] == None:
                cfg.sss[1] = len(obj)
            if cfg.sss[2] == None:
                cfg.sss[2] = 1

        if "start" in cfg.keys() or "stop" in cfg.keys() or "step" in cfg.keys():
            if cfg.start == None:
                cfg.start = 0
            if cfg.stop == None:
                cfg.stop = len(obj)
            if cfg.step == None:
                cfg.step = 1
    return cfg


def GDT(mobile, ref, sss=[None, None, None], cutoff=[0.5, 10, 0.5], true_resids=True, **kwargs):
    """
    GDT: Global Distance Test

    Algorithm to identify how good "mobile" matches to "reference" by calculating the sets
    of residues which do not deviate more than a specified (pairwise) CA distance cutoff.

    Note: hardcoded selection string and weights for alignment(GDT Algorithm)
        sel="protein and name CA"
        mobile_sel=mobile.select_atoms(sel)
        ref_sel=ref.select_atoms(sel)
        weights="mass"

    Args:
        mobile (MDA universe/atomgrp): mobile structure with trajectory
        ref (MDA universe/atomgrp): reference structure
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        cutoff (list): parameters describing GDT
            cutoff[0]: min_cutoff (float)
            cutoff[1]: max_cutoff (float)
            cutoff[2]: step_cutoff (float)
        true_resids (bool): offset resids of "GDT_resids" to correct codestyle resids into real resids
            True: return true resids in output "GDT_resids"
            False: return codestyle resids in output "GDT_resids"

    Kwargs:
        aliases for sss items:
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        aliases for cutoff items:
            min_cutoff (float)
            max_cutoff (float)
            step_cutoff (float)

    Returns:
        GDT_percent (list): GDT_Px for each cutoff distance in GDT_cutoff, where GDT_Px denotes
                            percent of residues under distance cutoff <= x Angstrom
        GDT_resids (list): resids within cutoff
            if true_resids == True:
                true resids; possible values are: minRES <= value <= maxRES
            if true_resids == False:
                codestyle resids; possible values are 0 <= value <= len(mobile.select_atoms(sel)))
        GDT_cutoff (list): cutoff distances in Angstrom
        RMSD (list): (RMSD before alignment, RMSD after elignment)
        FRAME (list): analyzed frames
    """
    ############################################################################
    # hardcoded selection string and weights for alignment (GDT Algorithm)
    sel = "protein and name CA"
    mobile_sel = mobile.select_atoms(sel)
    ref_sel = ref.select_atoms(sel)
    weights = "mass"
    ############################################################################
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "min_cutoff": cutoff[0],
               "max_cutoff": cutoff[1],
               "step_cutoff": cutoff[2]}
    cfg = _misc.CONFIG(default, **kwargs)
    cfg = _HELP_sss_None2int(mobile, cfg)  # convert None values of they keys sss, start, stop, step to integers

    FRAME = list(np.arange(cfg.start, cfg.stop, cfg.step))
    min_cutoff, max_cutoff, step_cutoff = cutoff[0], cutoff[1], cutoff[2]
    GDT_cutoff = list(np.arange(min_cutoff, max_cutoff+step_cutoff, step_cutoff))  # list with values 0.5 to 10 with steps of 0.5
    while GDT_cutoff[-1] > max_cutoff:
        GDT_cutoff = GDT_cutoff[: -1]
    GDT_resids = []  # list: RES indices (code-style) which fulfil cutoff condition
    GDT_percent = []  # list: percent of RES (CA) for each cutoff condition
    RMSD = []  # list: tuples with (RMSD before alignment, RMSD after alignment)

    # analyze trajectory
    for ts in tqdm(mobile.trajectory[cfg.start: cfg.stop: cfg.step]):
        PAIR_DISTANCES, _RMSD, _resids_mobile, _resids_ref = get_PairDistances(
            mobile, ref, sel=sel, mobile_sel=mobile_sel, ref_sel=ref_sel, weights=weights)
        RMSD.append(_RMSD)

        # get elements of PAIR_DISTANCES that are <= cutoff
        PD_ndx = []
        PD_percent = []
        shift = min(mobile_sel.residues.resids)
        for cutoff in GDT_cutoff:
            p, ndx = get_array_percent(PAIR_DISTANCES, cutoff)
            if true_resids:
                ndx += shift
            PD_ndx.append(ndx[0])
            PD_percent.append(p)

        GDT_resids.append(PD_ndx)
        GDT_percent.append(PD_percent)

    # test if RES ids within output match
    if np.any(_resids_mobile != _resids_ref):
        raise ValueError(f'''{GDT.__module__}.{GDT.__name__}():\
        \nGDT_resids: Residue IDs of mobile and reference don't match! Norm and align universe first.''')
    return(GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME)


def GDT_match_cutoff_dim(GDT_percent, GDT_cutoff):
    """
    Match GDT_cutoff array to the dimension of GDT_percent.
    Used in combination with analysis.PLOT() function to plot multiple curves on one canvas.

    Args:
        GDT_percent (list/array): output of analysis.GDT()
        GDT_cutoff  (list/array): output of analysis.GDT()

    Returns:
        GDT_cutoff (list/array): GDT_cutoff with same dimension as GDT_percent
    """
    GDT_cutoff = [GDT_cutoff for i in range(len(GDT_percent))]
    return GDT_cutoff


def plot_GDT(GDT_percent, GDT_cutoff, **kwargs):
    """
    Create a GDT_PLOT.

    Args:
        GDT_percent (list/array): output of analysis.GDT()
        GDT_cutoff (list/array): output of analysis.GDT()
        title (str)

    Kwargs:
        # see args of misc.figure()
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"figsize": (7.5, 5),
               "font_scale": 1.3,
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)

    plt_cutoff = GDT_match_cutoff_dim(GDT_percent, GDT_cutoff)
    fig, ax = PLOT(GDT_percent, plt_cutoff,
                   xlabel="Percent of residues", ylabel=r"Distance Cutoff ($\AA$)",
                   xlim=[0, 100], ylim=[0, 10], **cfg)
    return(fig, ax)


def get_GDT_TS(GDT_percent):
    """
    GDT_TS: GlobalDistanceTest_TotalScore
    Possible values: 0 <= GDT_TS <= 100

    Calculate GDT_TS for any given GDT_percent array according to:
        GDT_TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8)/4,
    where GDT_Px denotes the percentage of residues with distance cutoff <= x Angstrom

    Args:
        GDT_percent (list/array): output of analysis.GDT()

    Returns:
        GDT_TS (array)
    """
    GDT_TS = []
    for item in GDT_percent:
        score = (item[1] + item[3] + item[7] + item[15])/4
        GDT_TS.append(score)
    GDT_TS = np.array(GDT_TS)
    return GDT_TS


def get_GDT_HA(GDT_percent):
    """
    GDT_HA: GlobalDistanceTest_HighAccuracy
    Possible values: 0 <= GDT_HA <= 100

    Calculate GDT_HA for any given GDT_percent array according to:
        GDT_HA = (GDT_P0.5 + GDT_P1 + GDT_P2 + GDT_P4)/4,
    where GDT_Px denotes the percentage of residues with distance cutoff <= x Angstrom

    Args:
        GDT_percent (list/array): output of analysis.GDT()

    Returns:
        GDT_HA (array)
    """
    GDT_HA = []
    for item in GDT_percent:
        score = (item[0] + item[1] + item[3] + item[7])/4
        GDT_HA.append(score)
    GDT_HA = np.array(GDT_HA)
    return GDT_HA


def rank_scores(GDT_TS, GDT_HA, ranking_order="GDT_TS", prec=3, verbose=True):
    """
    Args:
        GDT_TS (array): GlobalDistanceTest_TotalScore
        GDT_HA (array): GlobalDistanceTest_HighAccuracy
        ranking_order (None/str):
            (None): output ranking ordered by frame number
            "FRAME": output ranking ordered by frame number
            "GDT_TS": output ranking ordered by GDT_TS
            "GDT_HA": output ranking ordered by GDT_HA
        prec (None/int):
            (None) or -1: rounding off
            (int):  rounding on to <prec> decimals

    Returns:
        GDT_TS_ranked (array): ranked array with GDT_TS scores
        GDT_HA_ranked (array): ranked array with GDT_HA scores
        GDT_ndx_ranked (array): corresponding element indices of scores
    """
    if ranking_order is None or ranking_order.upper() == "FRAME":
        GDT_TS_ranked = GDT_TS
        GDT_HA_ranked = GDT_HA
        GDT_ndx_ranked = np.array([ndx for ndx, i in enumerate(GDT_TS)])
        if verbose:
            print(f"Output ranking ordered by FRAME number")
    elif ranking_order.upper() == "GDT_TS":
        GDT_TS_ranked, GDT_ndx_ranked = _misc.get_ranked_array(GDT_TS, verbose=verbose)
        GDT_HA_ranked = np.array([GDT_HA[ndx] for ndx in GDT_ndx_ranked])
        if verbose:
            print(f"Output ranking ordered by GDT_TS")
    elif ranking_order.upper() == "GDT_HA":
        GDT_HA_ranked, GDT_ndx_ranked = _misc.get_ranked_array(GDT_HA, verbose=verbose)
        GDT_TS_ranked = np.array([GDT_TS[ndx] for ndx in GDT_ndx_ranked])
        if verbose:
            print(f"Output ranking ordered by GDT_HA")
    else:
        raise ValueError(f'''{rank_scores.__module__}.{rank_scores.__name__}():\
        \ninvalid value of 'ranking_order' parameter.''')

    if prec != None and prec != -1:
        GDT_TS_ranked = np.around(GDT_TS_ranked, prec)
        GDT_HA_ranked = np.around(GDT_HA_ranked, prec)

    return(GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked)


def GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", prec=3, verbose=True):
    """
    Args:
        GDT_percent (list): output of analysis.GDT()
        ranking_order (None/str):
            (None): output ranking orded by frame number
            "Frame": output ranking ordered by frame number
            "GDT_TS": output ranking ordered by GDT_TS
            "GDT_HA": output ranking ordered by GDT_HA
        prec (None/int):
            (None) or -1: rounding off
            (int): rounding to <prec> decimals

    Returns:
        GDT_TS_ranked (array): ranked array with GDT_TS values
        GDT_HA_ranked (array): ranked array with GDT_HA values
        GDT_ndx_ranked (array): array with corresponding index values

    Example:
        u = mda.Universe(<top>, <traj>)
        r = mda.universe(<ref> )
        GDT_cutoff, GDT_percent, GDT_resids, FRAME = analysis.GDT(u, r)
        GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = analysis.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS")
    """
    GDT_TS = get_GDT_TS(GDT_percent)
    GDT_HA = get_GDT_HA(GDT_percent)
    SCORES = rank_scores(GDT_TS, GDT_HA, ranking_order=ranking_order, prec=prec, verbose=verbose)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES
    return(GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked)


def GDT_rank_percent(GDT_percent):
    """
    Ranks GDT_percent based on the sum of GDT_Px for all x in cutoff.
    (Higher sum means better accuracy during protein alignment)

    Args:
        GDT_percent: output of analysis.GDT()

    Returns:
        RANKED_Psum (list): ranked percent sum
        RANKED_Psum_ndx (list): related indices of ranked percent sum
    """
    Psum = []
    for item in GDT_percent:
        Psum.append(sum(item))
    RANKED_Psum, RANKED_Psum_ndx = _misc.get_ranked_array(Psum, verbose=verbose)

    return(RANKED_Psum, RANKED_Psum_ndx)


def get_continuous_segments(array):
    """
    Note: single frame function

    Args:
        array (array): ordered array with integers representing resids ~ single frame
                       e.g.: array = [5, 6, 7, 12, 13, 18, 19, 20]

    Returns:
        SEGMENTS (list): list of continuous segments
    """
    SEGMENTS = []
    temp = []
    for i in range(len(array)):
        temp.append(array[i])
        try:
            if array[i]+1 != array[i+1]:
                SEGMENTS.append(temp)
                temp = []
        except IndexError:
            SEGMENTS.append(temp)

    return SEGMENTS


def GDT_continuous_segments(GDT_resids):
    """
    Note: multiple frame function

    Args:
        GDT_resids (array): output of analysis.GDT()

    Returns:
        SEGMENTS (list): list of continuous segments for each frame
    """
    SEGMENTS = []
    temp = []
    for frame in tqdm(GDT_resids):
        for array in frame:
            temp.append(get_continuous_segments(array))
        SEGMENTS.append(temp)
        temp = []

    return SEGMENTS


def plot_LA(mobile, ref, GDT_TS=[], GDT_HA=[], GDT_ndx=[], ndx_offset=0, rank_num=30,
            cmap="GDT_HA", show_cbar=True, show_frames=False, show_scores=True, save_as="", **kwargs):
    """
    Create LocalAccuracy Plot (heatmap) with:
        xdata = Residue ID
        ydata = Frame Number
        color = color-coded Pair Distance

    Note: be sure that you pass not too many data points otherwise this
          function will just squeeze them in.

    Args:
        mobile (MDA universe/atomgrp): mobile structure with trajectory
        ref (MDA universe/atomgrp): reference structure
        GDT_TS (array): array with GDT_TS values
        GDT_HA (array): array with GDT_HA values
        GTD_ndx (array): array with corresponding index values (representative for frame numbers)
        ndx_offset (int): offset/shift of GDT_ndx to match real "mobile" frames.
                          Look up "start" parameter during execution of analysis.GDT()
        rank_num (int): plot only <rank_num> best ranked frames
        cmap (str):
            "GDT_TS" or "TS": color map with new colors at values (0,  1, 2, 4, 8)
                              and vmin, vmax = (0, 10).
            "GDT_HA" or "HA": color map with new colors at values (0, .5, 1, 2, 4)
                              and vmin, vmax = (0, 5).
            other cmap names: see help(plt.colormaps) or alternatively
                              https://matplotlib.org/examples/color/colormaps_reference.html
        show_cbar (bool): show/hide colorbar
        show_scores (bool): show/hide frames
        show_scores (bool): show/hide GDT_TS and GDT_HA scores
        save_as (str): save name or realpath to save file

    Kwargs:
        # see args of misc.figure()
        prec (None/int): rounding of scores
            (None): rounding off
            (int):  rounding on
        cbar_label/label (str)
        cbar_fontweight/fontweight (str): "normal", "bold"
        cbar_location/location (str): "right", "bottom", "left", "top"
        cbar_orientation/orientation (str): "horizontal", "vertical"
        cbar_min / vmin (None/int): min value of colorbar and heatmap
        cbar_max / vmax (None/int): max value of colorbar and heatmap
        text_pos_Frame (list): [x0, y0] position of the "Frame" text box (label)
        text_pos_TS (list): [x0, y0] position of the "TS" text box (label)
        text_pos_HA (list): [x0, y0] position of the "HA" text box (label)

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
        LA_data (tuple):
            LA_data[0]: PairDistances (list)
            LA_data[1]: Frames (list)

    Example:
        # obtain data
        GDT = analysis.GDT(u, r, sss=[None,None,None])
        GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT

        # rank data
        SCORES = analysis.GDT_rank_scores(GDT_percent, ranking_order="GDT_HA")
        GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES

        # edit text box positions of labels "Frame", "TS", "HA"
        text_pos_kws = {"text_pos_Frame": [-8.8, -0.3],
                        "text_pos_TS": [-4.2, -0.3],
                        "text_pos_HA": [-1.9, -0.3]}

        # plot
        analysis.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2], **text_pos_kws)
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"figsize": (7.5, 6),
               "font_scale": 1.2,
               "prec": 2,
               "cmap": cmap,
               "cbar_label": r"mobile-reference CA-CA distances ($\AA$)",
               "cbar_fontweight": "bold",
               "cbar_location": 'right',
               "cbar_orientation": 'vertical',
               "cbar_min": None,
               "cbar_max": None,
               "text_pos_Frame": [-8.8, -0.3],
               "text_pos_TS": [-3.8, -0.3],
               "text_pos_HA": [-1.7, -0.3]}
    cfg = _misc.CONFIG(default, **kwargs)
    cfg.update_by_alias(alias="label", key="cbar_label", **kwargs)
    cfg.update_by_alias(alias="fontweight", key="cbar_fontweight", **kwargs)
    cfg.update_by_alias(alias="location", key="cbar_location", **kwargs)
    cfg.update_by_alias(alias="orientation", key="cbar_orientation", **kwargs)
    cfg.update_by_alias(alias="vmin", key="cbar_min", **kwargs)
    cfg.update_by_alias(alias="vmax", key="cbar_max", **kwargs)

    ############################################################################
    ### load data
    PAIR_DISTANCES = []
    FRAMES = [i+ndx_offset for i in GDT_ndx[:rank_num]]

    for ts in mobile.trajectory[FRAMES]:
        PD, *_ = get_PairDistances(mobile, ref)
        PAIR_DISTANCES.append(PD)

    if cfg.prec != None and cfg.prec != -1:
        GDT_TS = np.around(GDT_TS[: rank_num], cfg.prec)
        GDT_HA = np.around(GDT_HA[: rank_num], cfg.prec)

    xticks = mobile.residues.resids
    xticks = [x if x % 5 == 0 else "." for x in xticks]
    xticklabels = xticks

    if show_frames and show_scores:
        yticks = [f"{FRAMES[i]:>9}{GDT_TS[i]:>10.2f}{GDT_HA[i]:>8.2f} " if GDT_TS[i] != 100 else
                  f"{FRAMES[i]:>9}{GDT_TS[i]:>9.2f}{GDT_HA[i]:>8.2f} " for i in range(len(FRAMES))]
    elif show_frames:
        yticks = FRAMES
    elif show_scores:
        yticks = [f"{GDT_TS[i]:>10.2f}{GDT_HA[i]:>8.2f} " if GDT_TS[i] != 100 else
                  f"{GDT_TS[i]:>9.2f}{GDT_HA[i]:>8.2f} " for i in range(len(FRAMES))]

    yticklabels = yticks
    ############################################################################
    ### heatmap/cbar settings
    cmap_GDT = ["lightblue", "lightgreen", "yellow", "yellow", "orange", "orange",
                "orange", "orange", "red", "red"]

    if cfg.cmap == "GDT_HA" or cfg.cmap == "HA":
        vmin = 0.0
        vmax = 5.0
    elif cfg.cmap == "GDT_TS" or cfg.cmap == "TS":
        vmin = 0.0
        vmax = 10.0
    else:
        vmin = cfg.cbar_min
        vmax = cfg.cbar_max
    ############################################################################
    ### plot
    fig, ax = _misc.figure(**cfg)
    if show_cbar:
        cbar_ax = _misc.add_cbar_ax(ax, location=cfg.cbar_location,
                                    orientation=cfg.cbar_orientation)
        cbar_kws = {'orientation': cfg.cbar_orientation}
    else:
        cbar_ax = None
        cbar_kws = dict()

    if cfg.cmap == "GDT_HA" or cfg.cmap == "GDT_TS":
        hm = sns.heatmap(PAIR_DISTANCES, cmap=cmap_GDT, vmin=vmin, vmax=vmax,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         square=False, annot=False, linewidths=1.0,
                         ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, cbar=show_cbar)
    else:
        hm = sns.heatmap(PAIR_DISTANCES, cmap=cfg.cmap, vmin=vmin, vmax=vmax,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         square=False, annot=False, linewidths=1.0,
                         ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, cbar=show_cbar)
    if show_cbar:
        cbar = hm.collections[0].colorbar
        cbar.set_label(label=cfg.cbar_label, fontweight=cfg.cbar_fontweight)
        _misc.cbar_set_ticks_position(cbar, cfg.cbar_location)

    ax.tick_params(left=False, bottom=False)  # hide ticks of heatmap
    plt.title("Local Accuracy", fontweight='bold')
    plt.xlabel("Residue ID", fontweight='bold')

    # table labels
    if show_frames:
        ax.text(cfg.text_pos_Frame[0], cfg.text_pos_Frame[1], 'Frame', fontweight='bold')
    if show_scores:
        ax.text(cfg.text_pos_TS[0], cfg.text_pos_TS[1], 'TS', fontweight='bold')
        ax.text(cfg.text_pos_HA[0], cfg.text_pos_HA[1], 'HA', fontweight='bold')
    plt.tight_layout()
    plt.tight_layout()

    if save_as != "":
        _misc.savefig(save_as)

    if len(FRAMES) > 50:
        print("Displaying data for more than 50 frames...")
        print("Consider reducing the input data (e.g. rank scores and use top 40 frames).")

    LA_data = (PAIR_DISTANCES, FRAMES)
    return(fig, ax, LA_data)
