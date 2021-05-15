# @Author: Arthur Voronin
# @Date:   07.05.2021
# @Filename: gdt.py
# @Last modified by:   arthur
# @Last modified time: 15.05.2021


import pyrexMD.misc as _misc
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


def get_Pair_Distances(mobile, ref, sel1="protein and name CA", sel2="protein and name CA", **kwargs):
    """
    Aligns mobile to ref and calculates PairDistances (e.g. CA-CA distances).

    Args:
        mobile (MDA universe/atomgrp): mobile structure with trajectory
        ref (MDA universe/atomgrp): reference structure
        sel1 (str): selection string of mobile structure
        sel2 (str): selection string of reference structure

    Returns:
        PAIR_DISTANCES
        RMSD (tuple): (RMSD before alignment, RMSD after alignment)
        _resids_mobile (array)
        _resids_ref (array)
        """
    mobile_atoms = mobile.atoms.select_atoms(sel1)
    ref_atoms = ref.atoms.select_atoms(sel2)
    weights = "mass"  # hard coded weights
    #################################################

    # get_Pair_Distances function
    RMSD = _ana.alignto(mobile, ref, sel1=sel1, sel2=sel2, weights=weights)
    _resids_mobile, _resids_ref, PAIR_DISTANCES = mda.analysis.distances.dist(mobile_atoms, ref_atoms)
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


def GDT(mobile, ref, sel1="protein and name CA", sel2="protein and name CA",
        sss=[None, None, None], cutoff=[0.5, 10, 0.5], true_resids=True, **kwargs):
    """
    GDT: Global Distance Test

    Algorithm to identify how good "mobile" matches to "reference" by calculating the sets
    of residues which do not deviate more than a specified (pairwise) CA distance cutoff.

    Note:
        sel1 = mobile.select_atoms(sel1)
        sel2 = ref.select_atoms(sel2)
        weights="mass"  # hardcoded weights for alignment

    Args:
        mobile (MDA universe/atomgrp): mobile structure with trajectory
        ref (MDA universe/atomgrp): reference structure
        sel1 (str): selection string of mobile structure
        sel2 (str): selection string of reference structure
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
        disable (bool): disable progress bar

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
    #sel_mobile = mobile.select_atoms(sel1)
    #sel_ref = ref.select_atoms(sel2)
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


def GDT_match_cutoff_dim(GDT_percent, GDT_cutoff):
    """
    Match GDT_cutoff array to the dimension of GDT_percent.
    Used in combination with analysis.PLOT() function to plot multiple curves on one canvas.

    Args:
        GDT_percent (list/array): output of gdt.GDT()
        GDT_cutoff  (list/array): output of gdt.GDT()

    Returns:
        GDT_cutoff (list/array): GDT_cutoff with same dimension as GDT_percent
    """
    GDT_cutoff = [GDT_cutoff for i in range(len(GDT_percent))]
    return GDT_cutoff


def plot_GDT(GDT_percent, GDT_cutoff, **kwargs):
    """
    Create a GDT_PLOT.

    Args:
        GDT_percent (list/array): output of gdt.GDT()
        GDT_cutoff (list/array): output of gdt.GDT()
        title (str)

    Kwargs:
        # see args of misc.figure()

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
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
    GDT_TS: GlobalDistanceTest_TotalScore
    Possible values: 0 <= GDT_TS <= 100

    Calculate GDT_TS for any given GDT_percent array according to:
        GDT_TS = (GDT_P1 + GDT_P2 + GDT_P4 + GDT_P8)/4,
    where GDT_Px denotes the percentage of residues with distance cutoff <= x Angstrom

    Args:
        GDT_percent (list/array): output of gdt.GDT()

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
        verbose (bool)

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
        verbose (bool)

    Returns:
        GDT_TS_ranked (array): ranked array with GDT_TS values
        GDT_HA_ranked (array): ranked array with GDT_HA values
        GDT_ndx_ranked (array): array with corresponding index values

    Example:
        u = mda.Universe(<top>, <traj>)
        r = mda.universe(<ref> )
        GDT_cutoff, GDT_percent, GDT_resids, FRAME = gdt.GDT(u, r)
        GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS")
    """
    GDT_TS = get_GDT_TS(GDT_percent)
    GDT_HA = get_GDT_HA(GDT_percent)
    SCORES = rank_scores(GDT_TS, GDT_HA, ranking_order=ranking_order, prec=prec, verbose=verbose)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES
    return(GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked)


def GDT_rank_percent(GDT_percent, norm=True, verbose=False):
    """
    Ranks GDT_percent based on the sum of GDT_Px for all x in cutoff.
    (Higher sum means better accuracy during protein alignment)

    (Ranked) Psum can be view analogous to GDT scores but using all cutoffs
    instead of specific ones

    Args:
        GDT_percent (list): output of gdt.GDT()
        norm (bool): norms Psum during sum of GDT_Px
        verbose (bool)

    Returns:
        RANKED_Psum (list): ranked percent sum
        RANKED_Psum_ndx (list): related indices of ranked percent sum
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
    Note: single frame function

    Args:
        array (array): ordered array with integers representing resids ~ single frame
                       e.g.: array = [5, 6, 7, 12, 13, 18, 19, 20]

    Returns:
        SEGMENTS (list): list of continuous segments

    Example:
        >> gdt.get_continuous_segments([1,2,3,22,23,50,51,52])
        [[1, 2, 3], [22, 23], [50, 51, 52]]
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


def plot_LA(mobile, ref, GDT_TS=[], GDT_HA=[], GDT_ndx=[],
            sel1="protein and name CA", sel2="protein and name CA",
            ndx_offset=0, rank_num=30, cmap="GDT_HA",
            show_cbar=True, show_frames=False, show_scores=True,
            save_as="", **kwargs):
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
        sel1 (str): selection string of mobile structure (calculation of PairDistances)
        sel2 (str): selection string of reference structure (calculation of PairDistances)
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
        GDT = gdt.GDT(u, r, sss=[None,None,None])
        GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT

        # rank data
        SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_HA")
        GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES

        # edit text box positions of labels "Frame", "TS", "HA"
        text_pos_kws = {"text_pos_Frame": [-8.8, -0.3],
                        "text_pos_TS": [-4.2, -0.3],
                        "text_pos_HA": [-1.9, -0.3]}

        # plot
        gdt.plot_LA(mobile, ref, SCORES[0], SCORES[1], SCORES[2], **text_pos_kws)
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
        PD, *_ = get_Pair_Distances(mobile, ref, sel1=sel1, sel2=sel2)
        PAIR_DISTANCES.append(PD)

    if cfg.prec != None and cfg.prec != -1:
        GDT_TS = np.around(GDT_TS[: rank_num], cfg.prec)
        GDT_HA = np.around(GDT_HA[: rank_num], cfg.prec)

    xticks = mobile.select_atoms(sel1).residues.resids
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
        cbar_kws = dict()(7.5, 6)

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
