# @Author: Arthur Voronin
# @Date:   07.05.2021
# @Filename: contacts.py
# @Last modified by:   arthur
# @Last modified time: 13.06.2021

"""
This module contains functions related to native contact and bias contact analyses.


Example:
--------

.. code-block:: python

    import MDAnalysis as mda
    import pyrexMD.analysis.contacts as con

    ref = mda.Universe("path/to/pdb")
    mobile = mda.Universe("path/to/tpr", "path/to/xtc")

    # get and plot native contacts
    con.get_Native_Contacts(ref, sel="protein")
    con.plot_Contact_Map(ref, sel="protein")

    # compare contact map for n bias contacts
    n = 50
    con.plot_Contact_Map(ref, DCA_fin="path/to/bias/contacts", n_DCA=n)

    # check true positive rate for n bias contacts
    n = 50
    con.plot_DCA_TPR(ref, DCA_fin="path/to/bias/contacts", n_DCA=n)

Module contents:
----------------
"""

import pyrexMD.analysis.analysis as _ana
import pyrexMD.misc as _misc
import pyrexMD.topology as _top
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import contacts as _contacts, distances as _distances
import operator
from tqdm.notebook import tqdm


def get_Native_Contacts(ref, d_cutoff=6.0, sel="protein", **kwargs):
    """
    Get list of unique RES pairs and list of detailed RES pairs with native contacts.

    .. Note ::  len(NC) < len(NC_d) because NC contains only unique RES pairs
        whereas NC_d contains each ATOM pair.

    Args:
        ref (str, universe, atomgrp): reference structure
        d_cutoff (float): cutoff distance for native contacts
        sel (str): selection string

    Keyword Args:
        method (str):
          | '1' or 'Contact_Matrix': mda.contact_matrix() with d_cutoff
          | '2' or 'Shadow_Map' #TODO
        norm (bool): apply topology.norm_universe()
        ignh (bool): ignore hydrogen
        save_as (None, str):
          | save RES pairs and ATOM pairs of native contacts to a log file.
          | selection is hardcoded to "protein and name CA" for proteins
          | and "name N1 or name N3" for RNA/DNA.

    Returns:
        NC (list)
            native contacts with unique RES pairs
        NC_d (list)
            detailed list of NCs containing (RES pairs), (ATOM numbers), (ATOM names)
    """
    default = {"method": "1",
               "norm": True,
               "ignh": True,
               "save_as": None}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if type(cfg.method) is int:
        cfg.method = str(cfg.method)

    # load universe
    if type(ref) is str:
        u = mda.Universe(ref)
    else:
        u = ref
    if cfg.norm:
        _top.norm_universe(u)
    if cfg.ignh:
        a = _top.true_select_atoms(u, sel=sel, ignh=cfg.ignh, norm=cfg.norm)
    else:
        a = u.select_atoms(sel)

    NC = []     # list with unique NC pairs
    NC_d = []   # detailed list with NCs containing (RES pairs), (ATOM pairs), (NAME pairs)

    if cfg.method in ['1', 'Contact_Matrix', 'contact_matrix']:
        RES = a.resids
        IDS = a.ids
        NAMES = a.names

        CM = _distances.contact_matrix(a.positions, cutoff=d_cutoff)
        for i in range(len(CM)):
            for j in range(i + 1, len(CM)):  # no double count/reduce computing time
                if RES[j] - RES[i] > 3 and CM[i][j] == True and (RES[i], RES[j]) not in NC:
                    NC.append((RES[i], RES[j]))
                    NC_d.append([(RES[i], RES[j]), (IDS[i], IDS[j]), (NAMES[i], NAMES[j])])

    elif cfg.method in ['2', 'Shadow_Map', 'shadow_map']:
        # TODO method 2: SHADOWMAP
        pass

    # sort
    NC = sorted(NC)
    NC_d = sorted(NC_d)

    if cfg.save_as != None:
        cfg.save_as = _misc.realpath(cfg.save_as)
        _misc.cprint(f"Saved file as: {cfg.save_as}")
        if len(a.select_atoms("nucleic")) == 0:
            # hardcoded protein selection
            resid, resname, id, name = _top.parsePDB(u.filename, sel="protein and name CA", norm=cfg.norm)
        else:
            # hardcoded nucleic selection
            resid, resname, id, name = _top.parsePDB(u.filename, sel="name N1 or name N3", norm=cfg.norm)
        with open(cfg.save_as, "w") as fout:
            fout.write("#RESi\tRESj\tATOMi\tATOMj\n")
            for IJ in NC:
                fout.write(f"{IJ[0]}\t{IJ[1]}\t{id[resid.index(IJ[0])]}\t{id[resid.index(IJ[1])]}\n")

    return(NC, NC_d)


def get_NC_distances(mobile, ref, sss=[None, None, None], sel="protein", d_cutoff=6.0, plot=False, **kwargs):
    """
    get native contact distances.

    Args:
        mobile (universe, str)
        ref (universe, str)
        sel (str): selection string
        d_cutoff (float): cutoff distance
        plot (bool)

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): apply topology.norm_universe()
        ignh (bool): ignore hydrogen (mass < 1.2)
        save_as (None, str): save native contacts logfile as...

    Returns:
        NC (list)
            native contacts with unique RES pairs
        NC_dist (list)
            native contact distances ~ distances of items in NC
        DM (array)
            array of distance matrices
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True,
               "ignh": True,
               "save_as": None}
    cfg = _misc.CONFIG(default, **kwargs)
    if "save_as" in kwargs:
        del kwargs["save_as"]
    ##########################################################
    if isinstance(ref, str):
        ref = mda.Universe(ref)
    if isinstance(mobile, str):
        mobile = mda.Universe(mobile)
    if cfg.norm:
        _top.norm_universe(ref)
        _top.norm_universe(mobile)

    # convert sel to selids
    selid1 = _top.sel2selid(ref, sel=sel, norm=cfg.norm)
    selid2 = _top.sel2selid(mobile, sel=sel, norm=cfg.norm)

    # get native contacts and distance matrices
    NC, NC_d = get_Native_Contacts(ref, sel=selid1, d_cutoff=d_cutoff,
                                   ignh=cfg.ignh, save_as=cfg.save_as)
    DM = _ana.get_Distance_Matrices(mobile, sel=selid2, sss=[cfg.start, cfg.stop, cfg.step])

    # get native contact distances
    NC_dist = []
    for dm in DM:
        for item in NC:
            NC_dist.append(dm[item[0]-1][item[1]-1])

    if plot:
        _ana.plot_hist(NC_dist, **kwargs)
        plt.xlabel(r"Residue Pair Distance ($\AA$)", fontweight="bold")
        plt.ylabel(r"Count", fontweight="bold")
        plt.tight_layout()
    return (NC, NC_dist, DM)


def get_BC_distances(u, bc, sss=[None, None, None], sel="protein", plot=False, **kwargs):
    """
    get bias contact distances.

    Args:
        u (universe, str)
        bc (list): list of bias contacts represented by (RESi, RESj) tuple
        sel (str): selection string
        plot (bool)

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): apply topology.norm_universe()
        ignh (bool): ignore hydrogen (mass < 1.2)
        save_as (None, str): save native contacts logfile as...

    Returns:
        BC (list)
            bias contacts with unique RES pairs
        BC_dist (list)
            bias contact distances ~ distances of items in BC
        DM (array)
            array of distance matrices
    """
    BC = bc
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True,
               "ignh": True,
               "save_as": None}
    cfg = _misc.CONFIG(default, **kwargs)
    if "save_as" in kwargs:
        del kwargs["save_as"]
    ##########################################################
    if isinstance(u, str):
        u = mda.Universe(u)
    if cfg.norm:
        _top.norm_universe(u)

    # convert sel to selid
    selid = _top.sel2selid(u, sel=sel, norm=cfg.norm)

    # get native contacts and distance matrices
    DM = _ana.get_Distance_Matrices(u, sel=selid, sss=[cfg.start, cfg.stop, cfg.step])

    # get native contact distances
    BC_dist = []
    for dm in DM:
        for item in BC:
            BC_dist.append(dm[item[0]-1][item[1]-1])

    if plot:
        _ana.plot_hist(BC_dist, **kwargs)
        plt.xlabel(r"Residue Pair Distance ($\AA$)", fontweight="bold")
        plt.ylabel(r"Count", fontweight="bold")
        plt.tight_layout()
    return (BC, BC_dist, DM)


def plot_Contact_Map(ref, DCA_fin=None, n_DCA=None, d_cutoff=6.0,
                     sel='protein', pdbid='pdbid', **kwargs):
    """
    Create contact map based on the reference pdb structure.
    If DCA file is passed, correct/wrong contacts are visualized in green/red.

    Args:
        ref (str, universe, atomgrp): reference path or structure
        DCA_fin (None, str): DCA input file
        n_DCA (None, int): number of used DCA contacts
        d_cutoff (float): cutoff distance for native contacts
        sel (str): selection string
        pdbid (str): pdbid which is used for plot title

    Keyword Args:
        DCA_cols (tuple): columns containing the RES pairs in DCA_fin
        DCA_skiprows (int):
          | skip header rows of DCA_fin
          | -1 or "auto": auto detect
        filter_DCA (bool):
          | True: ignore DCA pairs with abs(i-j) < 3.
          | False: use all DCA pairs w/o applying filter.
        RES_range (None, list):
          | include RES only within [RES_min, RES_max] range
          | None: do not narrow down RES range
        ignh (bool): ignore hydrogen (mass < 1.2)
        norm (bool): apply topology.norm_universe()
        save_plot (bool)
        ms (None, int): marker size (should be divisible by 2)

    .. hint:: Args and Keyword Args of misc.figure() are also valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"DCA_cols": (0, 1),
               "DCA_skiprows": "auto",
               "filter_DCA": True,
               "RES_range": [None, None],
               "ignh": True,
               "norm": True,
               "save_plot": False,
               "ms": None}
    cfg = _misc.CONFIG(default, **kwargs)
    # init values
    if "figsize" not in kwargs:
        kwargs["figsize"] = (7, 7)

    # load universe
    if pdbid == 'pdbid':
        pdbid = _misc.get_PDBid(ref)
    if type(ref) is str:
        u = mda.Universe(ref)
    else:
        u = ref
    if cfg.norm:
        _top.norm_universe(u)
    if cfg.ignh:
        a = _top.true_select_atoms(u, sel, ignh=True, norm=cfg.norm)
    else:
        a = u.select_atoms(sel)
    res_min = min(a.resids)
    res_max = max(a.resids)

    # calculate nat. contacts for ref structure
    NC, NC_details = get_Native_Contacts(a, d_cutoff=d_cutoff, sel=sel, norm=cfg.norm)

    # PLOT
    fig, ax = _misc.figure(**kwargs)
    ax.set_aspect('equal')

    # conditions for markersize
    if cfg.ms is None and res_max - res_min >= 70:
        cfg.ms = 4
    elif cfg.ms is None and res_max - res_min <= 50:
        cfg.ms = 8
    elif cfg.ms is None:
        cfg.ms = 6

    print("Plotting native contacts...")
    for ndx, item in enumerate(NC):
        plt.scatter(item[0], item[1], color="silver", marker="s", s=cfg.ms*cfg.ms)

    # find matching contacts ij
    if DCA_fin is not None:
        DCA, _ = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=cfg.DCA_cols, skiprows=cfg.DCA_skiprows, filter_DCA=cfg.filter_DCA, RES_range=cfg.RES_range)
        for item in DCA:     # if item is in both lists plot it green otherwise red
            if item in NC:
                plt.plot(item[0], item[1], color="green", marker="s", ms=0.6*cfg.ms)
            else:
                plt.plot(item[0], item[1], color="red", marker="s", ms=0.6*cfg.ms)

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
    plt.xlabel(r"Residue i", fontweight="bold")
    plt.ylabel(r"Residue j", fontweight="bold")
    plt.title(f"Contact Map of {pdbid}", fontweight="bold")
    plt.tight_layout()
    if cfg.save_plot:
        _misc.savefig(filename=f"{pdbid}_Fig_Contact_Map.png", filedir="./plots")
    plt.show()
    return fig, ax


def plot_Contact_Map_Distances(ref, NC, NC_dist, pdbid="pdbid", **kwargs):
    """
    Create contact map with color-coded distances for selected contact pairs.

    Args:
        ref (universe, str): reference universe or pdb. Used to detect if target
          is protein or nucleic to select pre-defined color map.
        NC (list): output of get_NC_distances()
        NC_dist (list): output of get_NC_distances()
        pdbid (str): pdbid which is used for plot title

    Keyword Args:
        cmap (None, dict):
          | dictionary with "color":"threshold" pairs, e.g. "red":12.0
          | None: use pre-defined cmaps for Protein and RNA targets
        vmin (None, float)
        vmax (None, float)
        ms (None, int): marker size (should be divisible by 2)
        title (None, str): None, "default" or any title string
        save_plot (bool)

    .. hint:: Args and Keyword Args of misc.figure() and misc.add_cbar() are also valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    def __HELP__cmap2seq(cmap):
        """
        converts cmaps with "color":"threshold" items to cmap sequence as
        required by misc.create_cmap()
        """
        vmax = max(cmap.values())
        seq = []
        for v, k in cmap.items():
            seq.append(v)
            seq.append(k/vmax)
        del seq[-1]
        return seq
    ############################################################################
    cmap_Protein = {"dodgerblue": 2.0,
                    "lightgreen": 3.0,
                    "yellow": 4.0,
                    "orange": 5.0,
                    "red": 6.0}
    cmap_RNA = {"dodgerblue": 4.0,
                "lightgreen": 6.0,
                "yellow": 8.0,
                "orange": 10.0,
                "red": 12.0}
    default = {"cmap": None,
               "vmin": None,
               "vmax": None,
               "cbar_label": r"Residue Pair Distance ($\AA$)",
               "cbar_fontweight": "bold",
               "ms": None,
               "save_plot": False,
               "title": "default"}
    cfg = _misc.CONFIG(default, **kwargs)
    if isinstance(ref, str):
        ref = mda.Universe(ref)
    if cfg.cmap is None and len(ref.select_atoms("nucleic")) == 0:
        cfg.cmap = cmap_Protein
    elif cfg.cmap is None and len(ref.select_atoms("nucleic")) != 0:
        cfg.cmap = cmap_RNA
    # init values
    if "figsize" not in kwargs:
        kwargs["figsize"] = (7, 7)
    ############################################################################
    # PLOT
    fig, ax = _misc.figure(**kwargs)
    ax.set_aspect('equal')

    if cfg.vmax is None:
        cfg.vmax = max(cfg.cmap.values())
    _CMAP = _misc.create_cmap(__HELP__cmap2seq(cfg.cmap), vmin=cfg.vmin, vmax=cfg.vmax, ax=None)
    _CFG = cfg.deepcopy_without("cmap")
    _ = _misc.add_cbar(ax, cmap=_CMAP, **_CFG)

    # conditions for markersize
    I, J = _misc.unzip(NC)
    res_min = min(min(I), min(J))
    res_max = max(max(I), max(J))
    if cfg.ms is None and res_max - res_min >= 70:
        cfg.ms = 4
    elif cfg.ms is None and res_max - res_min <= 50:
        cfg.ms = 8
    elif cfg.ms is None:
        cfg.ms = 6

    #PLOT color-coded native contacts
    for ndx, item in enumerate(NC):
        for k, v in cfg.cmap.items():
            if NC_dist[ndx] <= v:
                plt.plot(item[0], item[1], color=k, marker="s", ms=0.6*cfg.ms)
                break
        # if NC distance is outside of cmap: color silver
        if NC_dist[ndx] > max(cfg.cmap.values()):
            plt.plot(item[0], item[1], color="silver", marker="s", ms=0.6*cfg.ms)

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
    plt.xlabel(r"Residue i", fontweight="bold")
    plt.ylabel(r"Residue j", fontweight="bold")
    if cfg.title == "default":
        plt.title(f"Contact Map of {pdbid}", fontweight="bold")
    else:
        plt.title(cfg.title, fontweight="bold")
    plt.tight_layout()
    if cfg.save_plot:
        _misc.savefig(filename=f"{pdbid}_Fig_Contact_Map.png", filedir="./plots")
    plt.show()
    return fig, ax


def plot_DCA_TPR(ref, DCA_fin, n_DCA, d_cutoff=6.0, sel='protein', pdbid='pdbid', **kwargs):
    """
    Plots true positive rate for number of used DCA contacts.

    - calculates shortest RES distance of the selection (only heavy atoms if ignh is True)
    - if distance is below threshold: DCA contact is True

    Args:
        ref (str): reference path
        ref (universe, atomgrp): reference structure
        DCA_fin (str): DCA input file (path)
        n_DCA (int): number of used DCA contacts
        d_cutoff (float): cutoff distance for native contacts
        sel (str): selection string
        pdbid (str): pdbid; used for plot title and figure name

    Keyword Args:
        DCA_cols (tuple, list): columns containing the RES pairs in DCA_fin
        DCA_skiprows (int):
          | skip header rows of DCA_fin
          | -1 or "auto": auto detect
        filter_DCA (bool):
          | True: ignore DCA pairs with abs(i-j) < 4
          | False: use all DCA pairs w/o applying filter
        RES_range (None):
          | [RES_min, RES_max] range. Ignore invalid RES ids
          | None: do not narrow down RES range
        ignh (bool): ignore hydrogen (mass < 1.2)
        norm (bool): apply topology.norm_universe()
        TPR_layer (str):
          | plot TPR curve in foreground or background layer
          | "fg", "foreground", "bg", "background"
        color (str)
        shade_area (bool): shade area between L/2 and L ranked contacts
        shade_color (str)
        shade_alpha (float)
        hline_color (None, str): color of 75p threshold (horizontal line)
        hline_ls (str): linestyle of hline (default: "-")
        nDCA_opt_color (None, str):
          | color of optimal/recommended nDCA threshold
          | nDCA_opt = misc.round_up(3/4*len(ref.residues), base=5))
          | (vertical line ~ nDCA_opt)
          | (horizontal line ~ TPR(nDCA_opt))
        nDCA_opt_ls (str): linestyle of nDCA_opt (default: "--")
        save_plot (bool)
        save_log (bool)
        figsize (tuple)
        marker (None, str): marker
        ms (int): markersize
        ls (str): linestyle
        alpha (float): alpha value

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"DCA_cols": (0, 1),
               "DCA_skiprows": "auto",
               "filter_DCA": True,
               "RES_range": [None, None],
               "ignh": True,
               "norm": True,
               "TPR_layer": "fg",
               "color": "blue",
               "shade_area": True,
               "shade_color": "orange",
               "shade_alpha": 0.2,
               "hline_color": "red",
               "hline_ls": "-",
               "nDCA_opt_color": "orange",
               "nDCA_opt_ls": "--",
               "save_plot": False,
               "save_log": False,
               "figsize": (8, 4.5),
               "marker": None,
               "ms": 1,
               "ls": "-",
               "alpha": 1.0}
    cfg = _misc.CONFIG(default, **kwargs)
    # load universe
    if pdbid == 'pdbid':
        pdbid = _misc.get_PDBid(ref)
    if isinstance(ref, str):
        u = mda.Universe(ref)
    else:
        u = ref
    if cfg.norm:
        _top.norm_universe(u)
    if cfg.ignh:
        a = _top.true_select_atoms(u, sel, ignh=True, norm=cfg.norm)
    else:
        a = u.select_atoms(sel)
    # read DCA and calculate TPR
    DCA, _ = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=cfg.DCA_cols, skiprows=cfg.DCA_skiprows, filter_DCA=cfg.filter_DCA, RES_range=cfg.RES_range)
    DCA_TPR = []  # List with TPR of DCA contacts with d < d_cutoff
    SD = _ana.shortest_RES_distances(u=a, sel=sel)[0]
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
        fig, ax = plt.subplots(figsize=cfg.figsize)
        if cfg.TPR_layer.lower() == "bg" or cfg.TPR_layer.lower() == "background":
            plt.plot(range(1, n_DCA + 1), DCA_TPR, color=cfg.color, marker=cfg.marker, ms=cfg.ms, ls=cfg.ls)
            plt.plot(range(1, n_DCA + 1), DCA_TPR, color=cfg.color, alpha=0.2, lw=2)
        # plt.legend(loc="upper right", numpoints=1)
        plt.xlim(0, n_DCA + 1)
        plt.ylim(0, 105)
        # if min(DCA_TPR) < 50:
        #     plt.ylim(0, 105)  # full range for TPR
        # else:
        #     plt.ylim(50, 105) # smaller range for very good TPR
        if cfg.shade_area:
            L1 = _misc.round_down(u.residues.n_residues/2, base=1)
            L2 = u.residues.n_residues
            plt.axvspan(L1, L2, alpha=cfg.shade_alpha, color=cfg.shade_color)
            plt.text(L1-2, -5, "L/2", color=cfg.shade_color, alpha=1, fontweight="bold")
            plt.text(L2, -5, "L", color=cfg.shade_color, alpha=1, fontweight="bold")
        if cfg.hline_color is not None:
            plt.axhline(75, color=cfg.hline_color, ls=cfg.hline_ls)
            plt.text(-3, 73, "75", color=cfg.hline_color, fontweight="bold")
            #_misc.set_pad(ax, xpad=None, ypad=10) # set padding after tight_layout()
        if cfg.nDCA_opt_color is not None:
            nDCA_opt = int(_misc.round_to_base(3/4*len(u.residues), 5))
            nDCA_opt_TPR = int(DCA_TPR[nDCA_opt - 1])  # requires index shift by 1 to get correct value
            plt.axhline(nDCA_opt_TPR, color=cfg.nDCA_opt_color, ls=cfg.nDCA_opt_ls)
            plt.text(-3, nDCA_opt_TPR-2, nDCA_opt_TPR, color=cfg.nDCA_opt_color, fontweight="bold")
            plt.axvline(nDCA_opt, color=cfg.nDCA_opt_color, ls=cfg.nDCA_opt_ls)
            plt.text(nDCA_opt-1.3, -5, nDCA_opt, color=cfg.nDCA_opt_color, fontweight="bold")
            #_misc.set_pad(ax, xpad=10, ypad=10)  # set padding after tight_layout()
        plt.xlabel("Number of ranked contacts")
        plt.ylabel("True Positive Rate (%)")
        plt.title(f"TPR of {pdbid}", fontweight="bold")
        sns.despine(offset=0)
        if cfg.TPR_layer.lower() == "fg" or cfg.TPR_layer.lower() == "foreground":
            plt.plot(range(1, n_DCA + 1), DCA_TPR, color=cfg.color, marker=cfg.marker, ms=cfg.ms, ls=cfg.ls)
            plt.plot(range(1, n_DCA + 1), DCA_TPR, color=cfg.color, alpha=0.2, lw=2)
        plt.tight_layout()
        if cfg.hline_color is not None:
            _misc.set_pad(ax, xpad=None, ypad=10)  # set padding after tight_layout()
        if cfg.nDCA_opt_color is not None:
            _misc.set_pad(ax, xpad=10, ypad=10)    # set padding after tight_layout()

        if cfg.save_plot:
            _misc.savefig(filename=f"{pdbid}_Fig_DCA_TPR.png", filedir="./plots")
        if cfg.save_log:
            save_dir = _misc.mkdir("./logs")
            file_dir = save_dir + f"/{pdbid}_DCA_TPR.log"
            with open(file_dir, "w") as fout:
                fout.write("Format:\n")
                fout.write("Number of DCA Contacts \t True Positive Rate (%)\n\n")
                fout.write("10 Best values (100% excluded):\n")
                count = 0
                for i in range(len(DCA_TPR_sorted)):
                    if count < 10 and DCA_TPR_sorted[i][1] < 100.0:
                        fout.write(f"{DCA_TPR_sorted[i][0]}\t{DCA_TPR_sorted[i][1]}\n")
                        count += 1
                fout.write("\nFull list:\n")
                fout.write(f"{0}\t{0.0}\n")
                for i in range(len(DCA_TPR)):
                    fout.write(f"{i+1}\t{DCA_TPR[i]}\n")
                print(f"Saved log as: {file_dir}")
    return fig, ax


def get_QNative(mobile, ref, sel="protein and name CA", sss=[None, None, None],
                d_cutoff=8.0, plot=True, verbose=True, **kwargs):
    """
    Get QValue for native contacts.

    - norms and aligns mobile to ref so that atomgroups have same resids
    - performs native contact analysis

    Args:
        mobile (universe): mobile structure with trajectory
        ref (universe): reference structure
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        d_cutoff (float): cutoff distance for native contacts
        plot (bool)
        verbose (bool)

    Keyword Arguments:
        sel1 (str): selection string for contacting group (1 to 1 mapping)
        sel2 (str): selection string for contacting group (1 to 1 mapping)
        method (str):
          | method for QValues calculation
          | 'radius_cut': native if d < d_cutoff
          | 'soft_cut': see help(MDAnalysis.analysis.contacts.soft_cut_q)
          | 'hardcut': native if d < d_ref
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        color (str): "r"
        alpha (float): 0.3
        marker (str): "."
        save_plot (bool)
        save_as (str): "QNative.png"

    .. Note :: sel (arg) is ignored if sel1 or sel2 (kwargs) are passed.

    Returns:
        FRAMES (list)
            list with frames
        QNATIVE (list)
            list with fraction of native contacts
    """
    default = {"sel1": None,
               "sel2": None,
               "method": "radius_cut",
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "color": "r",
               "alpha": 0.3,
               "marker": ".",
               "save_plot": False,
               "save_as": "QNative.png"}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.sel1 is None and cfg.sel2 is None:
        cfg.sel1 = sel
        cfg.sel2 = sel
    if cfg.sel1 == None or cfg.sel2 == None:
        _misc.Error("Set either sel (arg) or both sel1 and sel2 (kwargs).")
    #####################################################################
    _top.norm_and_align_universe(mobile, ref)

    ref1 = ref.select_atoms(cfg.sel1)  # reference group 1 in reference conformation
    ref2 = ref.select_atoms(cfg.sel2)  # reference group 2 in reference conformation

    results = _contacts.Contacts(mobile, select=(cfg.sel1, cfg.sel2), refgroup=(ref1, ref2),
                                 radius=d_cutoff, method=cfg.method)

    results.run(start=cfg.start, stop=cfg.stop, step=cfg.step, verbose=verbose)
    FRAMES = results.timeseries[:, 0]
    QNATIVE = results.timeseries[:, 1]

    if plot:
        _misc.cprint(f"average qnative value: {round(np.mean(QNATIVE), 3)}", "blue")

        fig, ax = _misc.figure()
        plt.plot(FRAMES, QNATIVE, color=cfg.color, alpha=cfg.alpha, marker=cfg.marker)
        plt.xlabel("Frame", fontweight="bold")
        plt.ylabel("QNative", fontweight="bold")
        plt.tight_layout()
    if cfg.save_plot:
        _misc.savefig(filename=cfg.save_as, create_dir=True)
    return FRAMES, QNATIVE


def get_QBias(mobile, bc, sss=[None, None, None], d_cutoff=8.0,
              prec=3, norm=True, plot=True, warn=True, verbose=True, **kwargs):
    """
    Get QValue for formed bias contacts.

    .. Note :: selection of get_QBias() is hardcoded to sel='protein and name CA'.

      Reason: bias contacts are unique RES PAIRS and grow 'slowly', but going from
      sel='protein and name CA' to sel='protein' increases the atom count
      significantly. Most of them will count as non-native and thus make
      the QBias value very low.

    .. Note :: MDAnalysis' qvalue algorithm includes selfcontacts. Comparison of
      both methods yields better results when the kwarg include_selfcontacts
      (bool) is set to True. However this improves the calculated QBias value
      artificially (e.g. even when all used bias contacts are never formed,
      QBias will not be zero due to the selfcontact counts)

    Args:
        mobile (universe): mobile structure with trajectory
        bc (list): list with bias contacts to analyze
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        d_cutoff (float): cutoff distance for native contacts
        prec (None, int): rounding precision
        norm (bool): norm universe before calculation.
        plot (bool)
        warn (bool): print important warnings about usage of this function.
        verbose (bool)

    Keyword Args:
        dtype (dtype): data type of returned contact matrix CM
        include_selfcontacts (bool):
          | sets norm of QBIAS. Default is False.
          | True: includes selfcontacts on main diagonal of CM (max count > len(bc))
          | False: ignores selfcontacts on main diagonal of CM (max count == len(bc))
        softcut (bool): Increase QBias if np.isclose(d, d_cutoff, atol=softcut_tol) is True. Defaults to True.
        softcut_tol (float): Softcut tolerance. Defatuls to 0.2 (Angstrom).
        softcut_inc: Increase QBias by this value if softcut applies. Defaults to 0.25.
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        figsize (tuple)
        color (str): "r"
        alpha (float): 0.3
        marker (str): "."
        save_plot (bool)
        save_as (str): "QBias.png"
        disable (bool): hide/show progress bar

    Returns:
        FRAMES (array)
            array with frame numbers
        QBIAS (array)
            array with fraction of formed bias contacts
        CM (array)
            array with distance matrices
    """
    def __HELP_softcut(d, d0, tol):
        """
        Returns value between 0 and 1, based on difference between d-d0 and d0+tol
        """
        if np.isclose(d, d0, atol=tol):
            return round(1.0-abs(d-d0)/tol, 3)
        return 0

    default = {"dtype": bool,
               "include_selfcontacts": False,
               "softcut": True,
               "softcut_tol": 0.2,
               "softcut_inc": 0.25,
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "figsize": (7, 5),
               "color": "r",
               "alpha": 0.3,
               "marker": ".",
               "save_plot": False,
               "save_as": "QBias.png",
               "disable": False}
    cfg = _misc.CONFIG(default, **kwargs)
    sel = "protein and name CA"
    ################################################################################
    if warn:
        _misc.cprint("Note 1: selection of get_QBias() is hardcoded to sel='protein and name CA'.", "red")
        _misc.cprint("Reason: bias contacts are unique RES PAIRS and grow 'slowly', but going from sel='protein and name CA' to sel='protein' increases the atom count significantly. Most of them will count as non-native and thus make the QBias value very low.", "red")
        _misc.cprint("Note 2: MDAnalysis' qvalue algorithm includes selfcontacts. Comparison of both methods yields better results when include_selfcontacts (bool, see kwargs) is set to True. However this improves the calculated QBias value artificially (e.g. even when all used bias contacts are never formed, QBias will not be zero due to the selfcontact counts)", "red")

    if norm:
        _top.norm_universe(mobile)
        a = mobile.select_atoms(sel)

    QBIAS = []  # QBias value (fraction of formed bias contacts)
    #CM = []     # Contact Matrices

    if verbose:
        _misc.cprint("calculating distance matrices...")
    DM = _ana.get_Distance_Matrices(mobile=mobile, sel=sel, sss=[cfg.start, cfg.stop, cfg.step], verbose=not cfg.disable)
    CM = [(dm <= d_cutoff) for dm in DM]  # converts distance matrix to bool matrix -> use as contact matrix

    if verbose:
        _misc.cprint("calculating QBias...")
    for i in tqdm(range(len(CM)), disable=cfg.disable):
        count = 0
        for item in bc:
            if CM[i][item[0]-1, item[1]-1] == True:
                count += 1
            elif cfg.softcut is True:
                d = DM[i][item[0]-1, item[1]-1]
                if (d-d_cutoff <= cfg.softcut_tol):
                    count += __HELP_softcut(d, d_cutoff, cfg.softcut_tol)
                    CM[i][item[0]-1, item[1]-1] = True
        if cfg.include_selfcontacts == False:
            # norm based on used bias contacts
            QBIAS.append(count/len(bc))
        else:
            # norm based on used bias contacts and selfcontacts
            QBIAS.append((count+len(a))/(len(bc)+len(a)))
    if prec is not None:
        QBIAS = [round(i, prec) for i in QBIAS]
    FRAMES = list(range(len(QBIAS)))

    if plot:
        if verbose:
            _misc.cprint(f"average QBias value: {np.mean(QBIAS)}", "blue")
        _cfg = cfg.deepcopy_without(["color", "alpha", "marker"])
        fig, ax = _ana.PLOT(xdata=FRAMES, ydata=QBIAS, color=cfg.color, alpha=cfg.alpha, marker=cfg.marker, **_cfg)
        plt.xlabel("Frame", fontweight="bold")
        plt.ylabel("QBias", fontweight="bold")
        plt.tight_layout()
    if cfg.save_plot:
        _misc.savefig(filename=cfg.save_as, create_dir=True)

    return np.array(FRAMES), np.array(QBIAS), np.array(CM)


def get_QBias_TPFP(mobile, BC, NC, sss=[None, None, None], d_cutoff=8.0, prec=3,
                   norm=True, plot=True, verbose=True, **kwargs):
    """
    Get QValue of true positive (TP) and false positive (FP) bias contacts.

    .. Note :: selection of this function is hardcoded to sel='protein and name CA'.

      Reason: bias contacts are unique RES PAIRS and grow 'slowly', but going from
      sel='protein and name CA' to sel='protein' increases the atom count
      significantly. Most of them will count as non-native and thus make
      the QBias value very low.

    .. Note :: MDAnalysis' qvalue algorithm includes selfcontacts. Comparison of
      both methods yields better results when the kwarg include_selfcontacts
      (bool) is set to True. However this improves the calculated QBias value
      artificially (e.g. even when all used bias contacts are never formed,
      QBias will not be zero due to the selfcontact counts)

    Args:
        mobile (universe): mobile structure with trajectory
        BC (list): list with bias contacts (saved as tuples (RESi, RESj))
        NC (list): list with native contacts (saved as tuples (RESi, RESj))
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        d_cutoff (float): cutoff distance for native contacts
        prec (None, int): rounding precision
        norm (bool): norm universe before calculation.
        plot (bool)
        verbose (bool)

    Keyword Args:
        dtype (dtype): data type of returned contact matrix CM
        include_selfcontacts (bool):
          | sets norm of QBIAS. Default is False.
          | True: includes selfcontacts on main diagonal of CM (max count > len(bc))
          | False: ignores selfcontacts on main diagonal of CM (max count == len(bc))
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        color_TP (str): color of true positive QBias. Defaults to "g".
        color_FP (str): color of false positive QBias. Defaults to "r".
        alpha (float): 0.3
        marker (str): "."
        disable (bool): hide/show progress bar

    Returns:
        FRAMES (array)
            array with frame numbers
        TP (array)
            array with Qvalue of true positive bias contacts
        FP (array)
            array with Qvalue of false positive bias contacts
        CM (array)
            array with distance matrices
    """
    default = {"dtype": bool,
               "include_selfcontacts": False,
               "start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "figsize": (7, 5),
               "color_TP": "g",
               "color_FP": "r",
               "alpha": 0.3,
               "marker": ".",
               "disable": False}
    cfg = _misc.CONFIG(default, **kwargs)
    sel = "protein and name CA"  # selection must be hardcoded (see print msg below)
    ################################################################################
    if not isinstance(BC, (list, np.ndarray)):
        raise _misc.Error("BC and NC must be list of tuples (RESi, RESj)")
    if not isinstance(NC, (list, np.ndarray)):
        raise _misc.Error("BC and NC must be list of tuples (RESi, RESj)")
    for item in BC:
        if not isinstance(item, tuple):
            raise _misc.Error("BC must be list of tuples (RESi, RESj)")
    for item in NC:
        if not isinstance(item, tuple):
            raise _misc.Error("NC must be list of tuples (RESi, RESj)")
    if norm:
        _top.norm_universe(mobile)

    CM = []  # Contact Matrices
    TP = []  # Qvalue of true positive bias contacts (fraction of formed contacts)
    FP = []  # Qvalue of false positive bias contacts (fraction of formed contacts)
    len_BC_TP = sum([True for item in BC if item in NC])
    len_BC_FP = sum([True for item in BC if item not in NC])
    TPR = round(100*(1-len_BC_FP/len_BC_TP), 2)
    if verbose:
        _misc.cprint(f"The used bias contacts have a TPR of {TPR}% with {len_BC_TP} native contacts and {len_BC_FP} non-native contacts.", "blue")

    # calculate distance matrices
    if verbose:
        _misc.cprint("Calculating Distance Matrices...")
    DM = _ana.get_Distance_Matrices(mobile=mobile, sel=sel, sss=[cfg.start, cfg.stop, cfg.step], verbose=verbose)
    for dm in DM:
        cm = (dm <= d_cutoff)   # converts distance matrix to bool matrix -> use as contact matrix
        CM.append(cm)

    # calculate TP and FP
    if verbose:
        _misc.cprint("Calculating QValues...")
    for cm in tqdm(CM, disable=cfg.disable):
        count_TP = 0
        count_FP = 0
        for item in BC:
            if item in NC and cm[item[0]-1, item[1]-1] == True:
                count_TP += 1
            elif item not in NC and cm[item[0]-1, item[1]-1] == True:
                count_FP += 1

        if cfg.include_selfcontacts == False:
            # norm based on used bias contacts
            TP.append(count_TP/len_BC_TP)
            FP.append(count_FP/len_BC_FP)
        else:
            # norm based on used bias contacts and selfcontacts
            TP.append((count_TP+len(cm))/(len_BC_TP+len(cm)))
            FP.append(count_FP/len_BC_FP)

    if prec is not None:
        TP = [round(i, prec) for i in TP]
        FP = [round(i, prec) for i in FP]
    FRAMES = range(len(TP))

    if plot:
        if verbose:
            _misc.cprint(f"average true positive QBias value: {np.mean(TP)}", "blue")
            _misc.cprint(f"average false positive QBias value: {np.mean(FP)}", "blue")
        _cfg = cfg.deepcopy_without(["color_TP", "color_FP"])
        fig, ax = _ana.PLOT(xdata=FRAMES, ydata=TP, color=cfg.color_TP, **_cfg)
        plt.xlabel("Frame", fontweight="bold")
        plt.ylabel("True Positive QBias", fontweight="bold")
        plt.tight_layout()
        fig, ax = _ana.PLOT(xdata=FRAMES, ydata=FP, color=cfg.color_FP, **_cfg)
        plt.xlabel("Frame", fontweight="bold")
        plt.ylabel("False Positive QBias", fontweight="bold")
        plt.tight_layout()
    return np.array(FRAMES), np.array(TP), np.array(FP), np.array(CM)


def get_formed_contactpairs(u, cm, sel="protein and name CA", norm=True, **kwargs):
    """
    get formed contact pairs by converting a contact matrix cm into a list with
    tuples (RESi,RESj).

    Args:
        u (universe): used to obtain topology information
        cm (array): single contact matrix of size (n_atoms, n_atoms)
        sel (str): selection string
        norm (bool): norm universe before obtaining topology information

    Keyword Args:
        include_selfcontacts (bool): include contacts on main diagonal of cm. Default is False.

    Returns:
        CP (list)
            list with contact pairs (RESi,RESj)
    """
    default = {"include_selfcontacts": False}
    cfg = _misc.CONFIG(default, **kwargs)
    ########################################################
    if norm:
        _top.norm_universe(u)
    RES = u.select_atoms(sel).resids   # get RESIDS information
    CP = []  # list with contact pairs

    ### test if input is single contact matrix
    if len(np.shape(cm)) == 3:
        raise _misc.Error("get_formed_contactpairs() expected contact map cm with size (n_atoms, n_atoms). You probably passed a contact map with size (n_frames, n_atoms, n_atoms)")

    ### calculate contact pairs
    if cfg.include_selfcontacts is False:
        # loop over upper triangle
        for i in range(len(cm)):
            for j in range(i+1, len(cm)):
                if cm[i][j] == True:
                    CP.append((RES[i], RES[j]))
    else:
        # loop over upper triangle including main diagonal
        for i in range(len(cm)):
            for j in range(i, len(cm)):
                if cm[i][j] == True:
                    CP.append((RES[i], RES[j]))
    return CP
