# @Author: Arthur Voronin
# @Date:   07.05.2021
# @Filename: gdt.py
# @Last modified by:   arthur
# @Last modified time: 22.05.2021


"""
This module contains functions related to the Global Distance Test.


Example:
--------

.. code-block:: python

    import MDAnalysis as mda
    import pyrexMD.misc as misc
    import pyrexMD.topology as top
    import pyrexMD.analysis.analysis as ana
    import pyrexMD.analysis.gdt as gdt

    ref = mda.Universe("<pdb_file>")
    mobile = mda.Universe("<tpr_file>", "<xtc_file>")

    # first norm and align universes
    top.norm_and_align_universe(mobile, ref)

    # run GDT
    GDT = gdt.GDT(mobile, ref)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT

    # get individual scores
    GDT_TS = gdt.get_GDT_TS(GDT_percent)
    GDT_HA = gdt.get_GDT_HA(GDT_percent)
    frames = [i for i in range(len(GDT_TS))]
    misc.cprint("GDT TS    GDT HA    frame", "blue")
    _ = misc.print_table([GDT_TS, GDT_HA, frames], verbose_stop=10, spacing=10)

    # rank scores
    SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", verbose=False)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES
    misc.cprint("GDT TS    GDT HA    frame", "blue")
    _ = misc.print_table([GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked], spacing=10, verbose_stop=10)


    # plot local accuracy
    text_pos_kws = {"text_pos_Frame": [-8.8, -0.3],
                    "text_pos_TS": [-14.2, -0.3],
                    "text_pos_HA": [-6, -0.3]}
    _ = gdt.plot_LA(mobile, ref, GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked, **text_pos_kws)

Module contents:
----------------
"""

import pyrexMD.misc as _misc
import pyrexMD.topology as _top
import pyrexMD.analysis.analysis as _ana
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

################################################################################
################################################################################
# GDT (Global Distance Test) Analyses


def get_array_percent(dist_array, cutoff):
    """
    Get percentage and indices of dist_array that fulfill: dist_array <= cutoff

    Args:
        dist_array (array): distance array
        cutoff (int, float): cutoff distance

    Returns:
        p (float)
            percentage of dist_array that fulfill: dist_array <= cutoff)
        ndx (array)
            indices of dist_array that fulfill: dist_array <= cutoff)
    """
    norm = len(dist_array)
    ndx = np.where(dist_array <= cutoff)
    p = _misc.percent(len(ndx[0]), norm)
    return(p, ndx)


def get_Pair_Distances(mobile, ref, sel1="protein and name CA", sel2="protein and name CA", **kwargs):
    """
    Aligns mobile to ref and calculates pair distances (e.g. CA-CA distances).

    .. Note:: single frame function.

    Args:
        mobile (universe, atomgrp): mobile structure with trajectory
        ref (universe, atomgrp): reference structure
        sel1 (str): selection string of mobile structure
        sel2 (str): selection string of reference structure

    Returns:
        PAIR_DISTANCES (array)
            array with pair pair distances
        RMSD (tuple)
            (RMSD before alignment, RMSD after alignment)
        _resids_mobile (array)
            array with mobile RES ids
        _resids_ref (array)
            array with reference RES ids
        """
    mobile_atoms = mobile.atoms.select_atoms(sel1)
    ref_atoms = ref.atoms.select_atoms(sel2)
    weights = "mass"  # hard coded weights
    #################################################
    RMSD = _ana.alignto(mobile, ref, sel1=sel1, sel2=sel2, weights=weights)
    _resids_mobile, _resids_ref, PAIR_DISTANCES = mda.analysis.distances.dist(mobile_atoms, ref_atoms)
    return(PAIR_DISTANCES, RMSD, _resids_mobile, _resids_ref)


def _HELP_sss_None2int(obj, cfg):
    """
    Converts None entries of the config keys to integers:
        sss = [None, None, None] -> [0, max frame of MDA universe, 1]
        start = None ->  0
        stop = None ->  max frame of u
        step = None ->  1

    Note:
        this function is (probably) required when a custom function uses either
        a MDA universe or (distance)matrix and a misc.CONFIG class with the keys
        "sss", "start", "stop", "step".

    Reason:
        slicing: aList[None:None:None] is equal to aList[::]
        get item: aList[None] does not exist


    Args:
        obj (universe, matrix): MDA universe or distance(matrix)
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


def GDT(mobile, ref, sel1="protein and name CA", sel2="protein and name CA",
        sss=[None, None, None], cutoff=[0.5, 10, 0.5], true_resids=True, **kwargs):
    """
    performs Global Distance Test (GDT).

    Algorithm to identify how good "mobile" matches to "reference" by calculating
    the sets of residues which do not deviate more than a specified (pairwise)
    CA distance cutoff.

    .. Note::
      | sel1 = mobile.select_atoms(sel1)
      | sel2 = ref.select_atoms(sel2)
      | weights="mass"  # hardcoded weights for alignment

    Args:
        mobile (universe, atomgrp): mobile structure with trajectory
        ref (universe, atomgrp): reference structure
        sel1 (str): selection string of mobile structure
        sel2 (str): selection string of reference structure
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        cutoff (list):
          | parameters describing GDT
          | cutoff[0]: min_cutoff (float)
          | cutoff[1]: max_cutoff (float)
          | cutoff[2]: step_cutoff (float)
        true_resids (bool):
          | offset resids of "GDT_resids" to correct codestyle resids into real resids
          | True: return true resids in output "GDT_resids"
          | False: return codestyle resids in output "GDT_resids"

    Keyword Args:
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        min_cutoff (float)
        max_cutoff (float)
        step_cutoff (float)
        disable (bool): disable progress bar

    Returns:
        GDT_percent (list)
          GDT_Px for each cutoff distance in GDT_cutoff, where GDT_Px denotes
          percent of residues under distance cutoff <= x Angstrom
        GDT_resids (list)
          | list with resids within cutoff
          | if true_resids == True:
          |   returns true resids; possible values are: minRES <= value <= maxRES
          | if true_resids == False:
          |   returns codestyle resids; possible values are 0 <= value <= len(mobile.select_atoms(sel)))
        GDT_cutoff (list)
          list with cutoff distances in Angstrom
        RMSD (list)
          list with (RMSD before alignment, RMSD after elignment)
        FRAME (list)
          list of analyzed frames
    """
    ############################################################################
    weights = "mass"  # hardcoded weights for alignment
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
    for ts in tqdm(mobile.trajectory[cfg.start: cfg.stop: cfg.step], **kwargs):
        PAIR_DISTANCES, _RMSD, _resids_mobile, _resids_ref = get_Pair_Distances(
            mobile, ref, sel1=sel1, sel2=sel2, weights=weights)
        RMSD.append(_RMSD)

        # get elements of PAIR_DISTANCES that are <= cutoff
        PD_ndx = []
        PD_percent = []
        shift = min(mobile.atoms.select_atoms(sel1).residues.resids)
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


def GDT_rna(mobile, ref, sel1="nucleic", sel2="nucleic", sss=[None, None, None],
            cutoff=[2, 40, 2], true_resids=True, **kwargs):
    """
    returns gdt.GDT() with RNA default values after converting selection strings
    to selection id strings.

    Code:
      | selid1 = pyrexMD.topology.sel2selid(mobile, sel=sel1)
      | selid2 = pyrexMD.topology.sel2selid(ref, sel=sel2)
      | return GDT(mobile=mobile, ref=ref, sel1=selid1, sel2=selid2, sss=sss,
      |            cutoff=cutoff, true_resids=true_resids, **kwargs)
    """
    selid1 = _top.sel2selid(mobile, sel=sel1)
    selid2 = _top.sel2selid(ref, sel=sel2)
    return GDT(mobile=mobile, ref=ref, sel1=selid1, sel2=selid2, sss=sss,
               cutoff=cutoff, true_resids=true_resids, **kwargs)


def GDT_match_cutoff_dim(GDT_percent, GDT_cutoff):
    """
    Matches GDT_cutoff array to the dimensions of GDT_percent. Used in combination
    with analysis.PLOT() function to plot multiple curves on one canvas.

    Args:
        GDT_percent (list, array): output of gdt.GDT()
        GDT_cutoff (list, array): output of gdt.GDT()

    Returns:
        GDT_cutoff (list, array)
            transformed GDT_cutoff with same dimensions as GDT_percent
    """
    GDT_cutoff = [GDT_cutoff for i in range(len(GDT_percent))]
    return GDT_cutoff


def plot_GDT(GDT_percent, GDT_cutoff, **kwargs):
    """
    Creates a GDT_PLOT.

    Args:
        GDT_percent (list, array): output of gdt.GDT()
        GDT_cutoff (list, array): output of gdt.GDT()
        title (str)

    .. Hint:: Args and Keyword Args of misc.figure() are valid Keyword Args.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"figsize": (7.5, 5),
               "font_scale": 1.3,
               "title": None}
    cfg = _misc.CONFIG(default, **kwargs)

    plt_cutoff = GDT_match_cutoff_dim(GDT_percent, GDT_cutoff)
    fig, ax = _ana.PLOT(GDT_percent, plt_cutoff,
                        xlabel="Percent of residues", ylabel=r"Distance Cutoff ($\AA$)",
                        xlim=[0, 100], ylim=[0, 10], **cfg)
    return(fig, ax)


def get_GDT_TS(GDT_percent):
    """
    .. Note:: GlobalDistanceTest_TotalScore. Possible values: 0 <= GDT_TS <= 100

    | Calculates GDT_TS for any given GDT_percent array according to:
    | GDT_TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8)/4,
    | where GDT_Px denotes the percentage of residues with distance cutoff <= x Angstrom

    Args:
        GDT_percent (list, array): output of gdt.GDT()

    Returns:
        GDT_TS (array)
            array with GDT_TS scores
    """
    GDT_TS = []
    for item in GDT_percent:
        score = (item[1] + item[3] + item[7] + item[15])/4
        GDT_TS.append(score)
    GDT_TS = np.array(GDT_TS)
    return GDT_TS


def get_GDT_HA(GDT_percent):
    """
    .. Note :: GlobalDistanceTest_HighAccuracy. Possible values: 0 <= GDT_HA <= 100

    | Calculate GDT_HA for any given GDT_percent array according to:
    | GDT_HA = (GDT_P0.5 + GDT_P1 + GDT_P2 + GDT_P4)/4,
    | where GDT_Px denotes the percentage of residues with distance cutoff <= x Angstrom

    Args:
        GDT_percent (list, array): output of gdt.GDT()

    Returns:
        GDT_HA (array)
            array with GDT_HA scores
    """
    GDT_HA = []
    for item in GDT_percent:
        score = (item[0] + item[1] + item[3] + item[7])/4
        GDT_HA.append(score)
    GDT_HA = np.array(GDT_HA)
    return GDT_HA


def rank_scores(GDT_TS, GDT_HA, ranking_order="GDT_TS", prec=3, verbose=True, **kwargs):
    """
    rank scores

    .. Note:: Similar function to gdt.GDT_rank_scores() but takes other arguments.

    Args:
        GDT_TS (array): output of gdt.get_GDT_TS()
        GDT_HA (array): output of gdt.get_GDT_HA()
        ranking_order (None, str):
          | "FRAME" or None: output ranking ordered by frame number
          | "GDT_TS": output ranking ordered by GDT_TS
          | "GDT_HA": output ranking ordered by GDT_HA
        prec (None, int):
          | -1 or None: rounding off
          | int: rounding on to this number of decimals
        verbose (bool)

    Keyword Args:
        verbose_stop (None, int): stop printing after N lines

    Returns:
        GDT_TS_ranked (array)
            ranked array with GDT_TS scores
        GDT_HA_ranked (array)
            ranked array with GDT_HA scores
        GDT_ndx_ranked (array)
            array with score indices for correct mapping
    """
    if ranking_order is None or ranking_order.upper() == "FRAME":
        GDT_TS_ranked = GDT_TS
        GDT_HA_ranked = GDT_HA
        GDT_ndx_ranked = np.array([ndx for ndx, i in enumerate(GDT_TS)])
        if verbose:
            print(f"Output ranking ordered by FRAME number")
    elif ranking_order.upper() == "GDT_TS":
        GDT_TS_ranked, GDT_ndx_ranked = _misc.get_ranked_array(GDT_TS, verbose=verbose, **kwargs)
        GDT_HA_ranked = np.array([GDT_HA[ndx] for ndx in GDT_ndx_ranked])
        if verbose:
            print(f"Output ranking ordered by GDT_TS")
    elif ranking_order.upper() == "GDT_HA":
        GDT_HA_ranked, GDT_ndx_ranked = _misc.get_ranked_array(GDT_HA, verbose=verbose, **kwargs)
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


def GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", prec=3, verbose=True, **kwargs):
    """
    rank scores.

    .. Note:: Similar function to gdt.rank_scores() but takes other arguments.

    Args:
        GDT_percent (list): output of gdt.GDT()
        ranking_order (None, str):
          | "FRAME" or None: output ranking ordered by frame number
          | "GDT_TS": output ranking ordered by GDT_TS
          | "GDT_HA": output ranking ordered by GDT_HA
        prec (None, int):
          | -1 or None: rounding off
          | int: rounding on to this number of decimals
        verbose (bool)

    Keyword Args:
        verbose_stop (None, int): stop printing after N lines

    Returns:
        GDT_TS_ranked (array)
            ranked array with GDT_TS scores
        GDT_HA_ranked (array)
            ranked array with GDT_HA scores
        GDT_ndx_ranked (array)
            array with score indices for correct mapping

    Example:
      | >> mobile = mda.Universe(<top>, <traj>)
      | >> ref = mda.universe(<ref> )
      | >> GDT_cutoff, GDT_percent, GDT_resids, FRAME = gdt.GDT(mobile, ref)
      | >> GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS")
    """
    GDT_TS = get_GDT_TS(GDT_percent)
    GDT_HA = get_GDT_HA(GDT_percent)
    SCORES = rank_scores(GDT_TS, GDT_HA, ranking_order=ranking_order, prec=prec,
                         verbose=verbose, **kwargs)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES
    return(GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked)


def GDT_rank_percent(GDT_percent, norm=True, verbose=False):
    """
    Ranks GDT_percent based on the sum of GDT_Px for all x in cutoff.
    (Higher sum means better accuracy during protein alignment)

    Ranked Psum can be view analogous to GDT scores but using all cutoffs
    instead of specific ones

    Args:
        GDT_percent (list): output of gdt.GDT()
        norm (bool): norms Psum ~ 1/N * sum of GDT_Px
        verbose (bool)

    Returns:
        RANKED_Psum (list)
            ranked percent sum
        RANKED_Psum_ndx (list)
            indices of ranked percent sum for correct mapping
    """
    Psum = []
    for item in GDT_percent:
        if norm:
            Psum.append(sum(item)/len(item))
        else:
            Psum.append(sum(item))
    RANKED_Psum, RANKED_Psum_ndx = _misc.get_ranked_array(Psum, verbose=verbose)

    return(RANKED_Psum, RANKED_Psum_ndx)


def get_continuous_segments(array):
    """
    Get continuous segments for single frame.

    Args:
        array (array):
          | ordered array with integers representing resids ~ single frame
          |  e.g.: array = [5, 6, 7, 12, 13, 18, 19, 20]

    Returns:
        SEGMENTS (list)
            list of continuous segments

    Example:
        | >> gdt.get_continuous_segments([1,2,3,22,23,50,51,52])
        | [[1, 2, 3], [22, 23], [50, 51, 52]]
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
    Get continous segments for multiple frames (due to GDT_resids)

    Args:
        GDT_resids (array): output of gdt.GDT()

    Returns:
        SEGMENTS (list)
            list of continuous segments for each frame
    """
    SEGMENTS = []
    temp = []
    for frame in tqdm(GDT_resids):
        for array in frame:
            temp.append(get_continuous_segments(array))
        SEGMENTS.append(temp)
        temp = []

    return SEGMENTS


def plot_LA(mobile, ref, GDT_TS, GDT_HA, GDT_ndx,
            sel1="protein and name CA", sel2="protein and name CA",
            cmap="GDT_HA", **kwargs):
    """
    Create LocalAccuracy Plot (heatmap) with

      - xdata = residue ID
      - ydata = frame number
      - color = color-coded pair distance

    .. Note:: do not pass too many data points otherwise the plot will get squeezed

    Args:
        mobile (universe, atomgrp): mobile structure with trajectory
        ref (universe, atomgrp): reference structure
        GDT_TS (array): array with GDT_TS scores.
        GDT_HA (array): array with GDT_HA scores.
        GTD_ndx (array): array with corresponding index values (representative for frame numbers).
        sel1 (str): selection string of mobile structure (calculation of pair distances)
        sel2 (str): selection string of reference structure (calculation of pair distances)
        cmap (str):
          | "GDT_TS" or "TS": color map with new colors at values (0,  1, 2, 4, 8)
            and vmin, vmax = (0, 10).
          | "GDT_HA" or "HA": color map with new colors at values (0, .5, 1, 2, 4)
            and vmin, vmax = (0, 5).
          | "nucleic" or "RNA" or "DNA": color map with new colors at values (0, .5, 1, 2, 4)
            and vmin, vmax = (0, 20).
          | other cmap names: see help(plt.colormaps) or alternatively
            https://matplotlib.org/examples/color/colormaps_reference.html


    Keyword Args:
        prec (None, int):
          | rounding precission of scores
          | None: rounding off
          | int:  rounding on to <int> decimals
        ndx_offset (int):
          | offset/shift of GDT_ndx to match real "mobile" frames. Defaults to 0.
          | Look up "start" parameter during execution of gdt.GDT()
        rank_num (int): plot only <rank_num> best ranked frames. Defaults to 30.
        show_cbar (bool): show/hide colorbar. Defaults to True.
        show_frames (bool): show/hide frame numbers. Defaults to False.
        show_scores (bool): show/hide GDT_TS and GDT_HA scores. Defaults to True.
        save_as (None, str): save name or realpath to save file. Defaults to None.
        cbar_ticks (None, list): color bar tick positions. Defaults to None.
        cbar_label/label (str)
        cbar_fontweight/fontweight (str): "normal", "bold"
        cbar_location/location (str): "right", "bottom", "left", "top"
        cbar_orientation/orientation (str): "horizontal", "vertical"
        cbar_min/vmin (None, int): min value of colorbar and heatmap. Gets
          overwritten by cmaps such as "GDT_TS", "GDT_HA", "RNA" etc.
        cbar_max/vmax (None, int): max value of colorbar and heatmap. Gets
          overwritten by cmaps such as "GDT_TS", "GDT_HA", "RNA" etc.
        text_pos_Frame (list): [x0, y0] position of the "Frame" text box (label)
        text_pos_TS (list): [x0, y0] position of the "TS" text box (label)
        text_pos_HA (list): [x0, y0] position of the "HA" text box (label)
        font_scale (float)

    .. Hint:: Args and Keyword of misc.figure() are also valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
        LA_data (tuple)
            | LA_data[0]: PairDistances (list)
            | LA_data[1]: Frames (list)
    Example:
        | # obtain data
        | >> GDT = gdt.GDT(mobile, ref, sss=[None,None,None])
        | >> GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT
        |
        | # rank data
        | >> SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_HA")
        | >> GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES
        |
        | # edit text box positions of labels "Frame", "TS", "HA"
        | >>text_pos_kws = {"text_pos_Frame": [-8.8, -0.3],
        |                   "text_pos_TS": [-4.2, -0.3],
        |                   "text_pos_HA": [-1.9, -0.3]}
        |
        | # plot
        | >> gdt.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2], **text_pos_kws)
    """
    # init CONFIG object with default parameter and overwrite them if kwargs contain the same keywords.
    default = {"figsize": (7.5, 6),
               "font_scale": 1.2,
               "ndx_offset": 0,
               "rank_num": 30,
               "show_cbar": True,
               "show_frames": False,
               "show_scores": True,
               "save_as": None,
               "prec": 2,
               "cmap": cmap,
               "cbar_ticks": None,
               "cbar_label": r"mobile-reference CA-CA distances ($\AA$)",
               "cbar_fontweight": "bold",
               "cbar_location": 'right',
               "cbar_orientation": 'vertical',
               "cbar_min": None,
               "cbar_max": None,
               "vmin": None,
               "vmax": None,
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
    FRAMES = [i+cfg.ndx_offset for i in GDT_ndx[:cfg.rank_num]]

    for ts in mobile.trajectory[FRAMES]:
        PD, *_ = get_Pair_Distances(mobile, ref, sel1=sel1, sel2=sel2)
        PAIR_DISTANCES.append(PD)

    if cfg.prec != None and cfg.prec != -1:
        GDT_TS = np.around(GDT_TS[: cfg.rank_num], cfg.prec)
        GDT_HA = np.around(GDT_HA[: cfg.rank_num], cfg.prec)

    xticks = mobile.select_atoms(sel1).residues.resids
    xticks = [x if x % 5 == 0 else "." for x in xticks]
    xticklabels = xticks

    if cfg.show_frames and cfg.show_scores:
        yticks = [f"{FRAMES[i]:>9}{GDT_TS[i]:>10.2f}{GDT_HA[i]:>8.2f} " if GDT_TS[i] != 100 else
                  f"{FRAMES[i]:>9}{GDT_TS[i]:>9.2f}{GDT_HA[i]:>8.2f} " for i in range(len(FRAMES))]
    elif cfg.show_frames:
        yticks = FRAMES
    elif cfg.show_scores:
        yticks = [f"{GDT_TS[i]:>10.2f}{GDT_HA[i]:>8.2f} " if GDT_TS[i] != 100 else
                  f"{GDT_TS[i]:>9.2f}{GDT_HA[i]:>8.2f} " for i in range(len(FRAMES))]

    yticklabels = yticks
    ############################################################################
    ### heatmap/cbar settings
    cmap_GDT = ["lightblue", "lightgreen", "yellow", "yellow", "orange", "orange",
                "orange", "orange", "red", "red"]
    cmap_RNA = ["lightblue", "lightblue", "lightgreen", "lightgreen",
                "yellow", "yellow", "orange", "orange", "red", "red"]

    # apply color bar limits if passed (vmin and vmax have higher priority than cbar_min and cbar_max)
    if cfg.cbar_min is not None:
        cfg.vmin = cfg.cbar_min
    if cfg.cbar_max is not None:
        cfg.vmax = cfg.cbar_max
    # if no limits passed: apply pre-defined limits
    if cfg.cmap in ["GDT_HA", "HA"]:
        if cfg.vmin is None:
            cfg.vmin = 0.0
        if cfg.vmax is None:
            cfg.vmax = 5.0
    elif cfg.cmap in ["GDT_TS", "TS"]:
        if cfg.vmin is None:
            cfg.vmin = 0.0
        if cfg.vmax is None:
            cfg.vmax = 10.0
    elif cfg.cmap in ["nucleic", "rna", "dna", "RNA", "DNA"]:
        if cfg.vmin is None:
            cfg.vmin = 0.0
        if cfg.vmax is None:
            cfg.vmax = 20.0
    ############################################################################
    ### plot
    fig, ax = _misc.figure(**cfg)
    if cfg.show_cbar:
        cbar_ax = _misc.add_cbar_ax(ax, location=cfg.cbar_location,
                                    orientation=cfg.cbar_orientation)
        cbar_kws = {'orientation': cfg.cbar_orientation}
    else:
        cbar_ax = None
        cbar_kws = dict()(7.5, 6)

    if cfg.cmap in ["GDT_TS", "TS", "GDT_HA", "HA"]:
        hm = sns.heatmap(PAIR_DISTANCES, cmap=cmap_GDT, vmin=cfg.vmin, vmax=cfg.vmax,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         square=False, annot=False, linewidths=1.0,
                         ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, cbar=cfg.show_cbar)
    if cfg.cmap in ["nucleic", "rna", "dna", "RNA", "DNA"]:
        hm = sns.heatmap(PAIR_DISTANCES, cmap=cmap_RNA, vmin=cfg.vmin, vmax=cfg.vmax,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         square=False, annot=False, linewidths=1.0,
                         ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, cbar=cfg.show_cbar)
    else:
        hm = sns.heatmap(PAIR_DISTANCES, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         square=False, annot=False, linewidths=1.0,
                         ax=ax, cbar_ax=cbar_ax, cbar_kws=cbar_kws, cbar=cfg.show_cbar)
    if cfg.show_cbar:
        cbar = hm.collections[0].colorbar
        cbar.set_label(label=cfg.cbar_label, fontweight=cfg.cbar_fontweight)
        _misc.cbar_set_ticks_position(cbar, cfg.cbar_location)
        if cfg.cbar_ticks is None and cfg.cmap in ["nucleic", "rna", "dna", "RNA", "DNA"]:
            cbar.set_ticks(np.arange(0, 22, 2))
        if cfg.cbar_ticks is not None:
            cbar.set_ticks(cfg.cbar_ticks)

    ax.tick_params(left=False, bottom=False)  # hide ticks of heatmap
    plt.title("Local Accuracy", fontweight='bold')
    plt.xlabel("Residue ID", fontweight='bold')

    # table labels
    if cfg.show_frames:
        ax.text(cfg.text_pos_Frame[0], cfg.text_pos_Frame[1], 'Frame', fontweight='bold')
    if cfg.show_scores:
        ax.text(cfg.text_pos_TS[0], cfg.text_pos_TS[1], 'TS', fontweight='bold')
        ax.text(cfg.text_pos_HA[0], cfg.text_pos_HA[1], 'HA', fontweight='bold')
    plt.tight_layout()
    plt.tight_layout()

    if cfg.save_as != None:
        _misc.savefig(cfg.save_as)

    if len(FRAMES) > 50:
        print("Displaying data for more than 50 frames...")
        print("Consider reducing the input data (e.g. rank scores and use top 40 frames).")

    LA_data = (PAIR_DISTANCES, FRAMES)
    return(fig, ax, LA_data)


def plot_LA_rna(mobile, ref, GDT_TS, GDT_HA, GDT_ndx,
                sel1="nucleic", sel2="nucleic", cmap="RNA", **kwargs):
    """
    plot_LA() with RNA default values after converting selection strings to selection id strings.

    Code:
      | selid1 = pyrexMD.topology.sel2selid(mobile, sel=sel1)
      | selid2 = pyrexMD.topology.sel2selid(ref, sel=sel2)
      | return plot_LA(mobile, ref, GDT_TS=GDT_TS, GDT_HA=GDT_HA, GDT_ndx=GDT_ndx,
      |               sel1=selid1, sel2=selid2, cmap=cmap, **kwargs)
    """
    selid1 = _top.sel2selid(mobile, sel=sel1)
    selid2 = _top.sel2selid(ref, sel=sel2)
    return plot_LA(mobile, ref, GDT_TS=GDT_TS, GDT_HA=GDT_HA, GDT_ndx=GDT_ndx,
                   sel1=selid1, sel2=selid2, cmap=cmap, **kwargs)
