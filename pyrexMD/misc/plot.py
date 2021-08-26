# @Author: Arthur Voronin <arthur>
# @Date:   26.08.2021
# @Filename: plot.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

"""
.. hint:: This module is a collection of plot-related functions which are used
    frequently. Included functions may contain modified versions of small existing
    functions to extend their default behavior in order to streamline pyrexMD.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pl

from pyrexMD.misc.classes import CONFIG
from pyrexMD.misc.func import joinpath, round_up, round_down, flatten_array

################################################################################
################################################################################
### custom plt.figure()


def figure(num=None, figsize=(6.5, 4), dpi=None, grid=[1, 1], hr=[], wr=[],
           palette=None, n_colors=None,
           context='notebook', style='ticks', font_scale=0,
           despine=True, offset=0, **kwargs):
    """
    Top level container for all the plot elements. Modified version of plt.figure()
    with gridspec and seaborn elements. "Example" below shows how to plot data
    by using misc.figure() and plt.sca() # set current axis

    Args:
        num (int): figure.number (to reuse the figure canvas)
        figsize (tuple)
        dpi (None, int):
          | dpi settings of figure
          | if monitor dpi is 100 and figure dpi is 300, then figure will be displayed
          | on the monitor 3x larger (compared to printed version with e.g. 300 dpi printer).
        grid (list): nrows, ncols ~ height x width
        hr (list): height ratio of rows
        wr (list) width ratio of columns
        font_scale (float)
        palette (None, str, list, tuple):
          | None: use current palette
          | str: name of the palette, see seaborn documentation
          | list: list of colors
          | tuple: tuple of colors
        n_colors (None, int):
          | None: use default number of color cycles based on palette
          | int: remap number of color cycles for the palette
        context (None, dict, str):
          | Affects sizes and widths of the plot, e.g. labelsize,
          | linesize, etc. See return values of sns.plotting_context() to get
          | all valid dict settings.
          | str:'notebook': scaled by 1.0
          |     'paper': scaled by 0.8
          |     'talk': scaled by 1.3
          |     'poster': scaled by 1.6
        style (None, dict, str):
          | Affects aesthetic style of the plots, e.g. axes color,
          | grid display, etc. See return values of sns.axes_style() to get all
          | valid dict settings.
          | str:'ticks', 'dark', 'white', 'darkgrid', 'whitegrid'
        despine (bool):
          | Remove the top and right spines from all axes. See sns.despine()
          | documentation for individual axis spine removal.
        offset (dict/int):
          | Absolute distance of spines. Use a dict to offset individual spines,
          | e.g. offset={"bottom":0, "left":20}.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes

    Example:
        | fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        | plt.sca(ax[0])
        | plt.plot([0,1,2], [0,4,4])
        | plt.plot([0,1,2], [0,10,10])
        | plt.sca(ax[2])
        | plt.plot([0,-1,-2], [0,-4,-4])
        | plt.plot([0,-1,-2], [0,-10,-10])
    """
    if grid[0] == 1:
        hr = [1]
    if grid[1] == 1:
        wr = [1]

    if grid != [1, 1] and hr == []:
        hr = [1] * grid[0]
    if grid != [1, 1] and wr == []:
        wr = [1] * grid[1]

    if font_scale == 0:
        sns.set(context=context, style=style)  # context scales font elements by default
    else:
        sns.set(context=context, style=style, font_scale=font_scale)  # double scaling with font_scale
    sns.set_palette(palette, n_colors)

    if num is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    else:
        fig = plt.figure(num=num, figsize=figsize)

    gs = matplotlib.gridspec.GridSpec(grid[0], grid[1], width_ratios=wr, height_ratios=hr)
    ax = []

    for gs_item in gs:
        ax.append(plt.subplot(gs_item))

    if despine == True:
        sns.despine(offset=offset)

    plt.tight_layout()
    #plt.show()

    if len(ax) == 1:
        ax = ax[0]
    return (fig, ax)


def scatter(x, y, z=None, **kwargs):
    """
    Creates scatter plot. Applies cmap for z values if z is passed.

    Args:
        x (array)
        y (array)
        z (None, array)

    Keyword Args:
        figsize (tuple): Defaults to (6.6, 5.6)
        aspect ('auto', 'equal', int):
          | aspect ratio of figure. Defaults to 'auto'.
          | 'auto': fill the position rectangle with data.
          | 'equal': synonym for aspect=1, i.e. same scaling for x and y.
          | int: a circle will be stretched such that the height is *int* times the width.
        marker (None, str)
        ms (None, int): marker size
        cmap (str):
          | colormap name, e.g. 'virids', 'plasma', 'inferno', 'summer', 'winter', 'cool', etc.
          | You can reverse the cmap by appending '_r' to the name.
          | See https://matplotlib.org/stable/tutorials/colors/colormaps.html
        vmin (None, float): min value of cmap and colorbar
        vmax (None, float): max value of cmap and colorbar
        cbar_label (None, str)
        xlabel (None, str)
        ylabel (None, str)

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"figsize": (6.6, 5.6),
               "aspect": "auto",
               "marker": None,
               "ms": None,
               "cmap": "viridis",
               "vmin": None,
               "vmax": None,
               "cbar_label": None,
               "xlabel": None,
               "ylabel": None}
    cfg = CONFIG(default, **kwargs)
    ###########################################################
    fig, ax = figure(**cfg)
    plt.scatter(x=x, y=y, c=z, s=cfg.ms, marker=cfg.marker, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax)
    plt.xlabel(cfg.xlabel, fontweight="bold")
    plt.ylabel(cfg.ylabel, fontweight="bold")
    add_cbar(ax=ax, **cfg)
    ax.set_aspect(cfg.aspect)
    plt.tight_layout()
    return (fig, ax)


def set_pad(fig_or_ax, xpad=None, ypad=None):
    """
    Set pad (spacing) between axis and axis labels

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        xpad (None, float):
          | spacing between xaxis and its labels
          | None: use current settings
        ypad (None, float):
          | spacing between yaxis and its labels
          | None: use current settings
    """
    default = {"xpad": 3.5,
               "ypad": 3.5}
    cfg = CONFIG(default, xpad=xpad, ypad=ypad)
    ############################################################################
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        if len(fig_or_ax.axes) == 1:
            ax = [fig_or_ax.axes]
        else:
            ax = fig_or_ax.axes
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        ax = [fig_or_ax]
    elif isinstance(fig_or_ax, list):
        ax = fig_or_ax

    for ax in ax:
        if xpad is not None:
            for tick in ax.get_xaxis().get_major_ticks():
                tick.set_pad(cfg.xpad)
        if ypad is not None:
            for tick in ax.get_yaxis().get_major_ticks():
                tick.set_pad(cfg.ypad)
    return


def legend(labels=[""], handlecolors=[""], handlelength=1, handletextpad=None, loc=None, **kwargs):
    """
    Alias function of plt.legend() with most frequently used parameters.

    Args:
        labels (sequence of strings)
        handlescolors (list)
        handlelength (None/int/float)
        handletextpad (None/int/float)
        loc (str):
          | Location String   Location Code
          | -------------------------------
          | 'best'            0
          | 'upper right'     1
          | 'upper left'      2
          | 'lower left'      3
          | 'lower right'     4
          | 'right'           5
          | 'center left'     6
          | 'center right'    7
          | 'lower center'    8
          | 'upper center'    9
          | 'center'          10

    Keyword Args:
        title (str)
        edgecolor (str)
        fancybox (bool):
          | True:  legendbox with round edges
          | False: legendbox with normal edges
    """
    legend = plt.legend(labels=labels,
                        handlelength=handlelength,
                        handletextpad=handletextpad,
                        loc=loc, **kwargs)
    if handlecolors != [""]:
        for ndx, color in enumerate(handlecolors):
            legend.legendHandles[ndx].set_color(color)
    return


def savefig(filename, filedir="", create_dir=True, dpi=300):
    """
    general helpfunction

      - save current figure
      - print message "Saved figure as: ..."

    Args:
        filename (None, str):
          | None: do not save figure
          | str: file name or realpath to file
        filedir (str): file directory
        create_dir (bool): create directory if it does not exist yet
        dpi (int): dpi settings

    Returns:
        realpath (str)
            realpath to saved figure
    """
    if filename == "" or filename is None:
        return
    if filedir != "":
        realpath = joinpath(filedir, filename, create_dir=create_dir)
    else:
        realpath = joinpath("./", filename, create_dir=create_dir)
    plt.savefig(realpath, dpi=dpi)
    print("Saved figure as:", realpath)
    return realpath


### alias function
save_fig = savefig


def autoapply_limits(fig_or_ax, margin=0.05):
    """
    Apply plt.xlim() and plt.ylim() on each axis object based on its xmin, xmax,
    ymin, ymax values.

    .. NOTE::

      - use only after plotting data
      - currently works only with Line2D data

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        margin (float): extra margin on upper limits, where margin=0.01 equals 1% of xmax and ymax

    Returns:
        xlim (list)
            if fig_or_ax is axis
        ylim (list)
            if fig_or_ax is axis
        XLIM (list of lists)
            if fig_or_ax is figure
        YLIM (list of lists)
            if fig_or_ax is figure

    Example:
        | >> fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        | >> plt.sca(ax[0])
        | >> plt.plot([0,1,2], [0,4,4])
        | >> plt.plot([0,1,2], [0,10,10])
        | >> autoapply_limits(fig, margin=0.05)
        | ([[0, 2.1], [0, 0], [0, 0], [0, 0]],
        | [[0,10.5], [0, 0], [0, 0], [0, 0]])
    """
    # Test if object is figure or axis
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        axes = fig_or_ax.get_axes()
    if "axes" in locals():
        pass
    else:
        axes = [fig_or_ax]

    # get and apply limits
    XLIM = []
    YLIM = []

    for ax in axes:
        xlim = [0, 0]
        ylim = [0, 0]
        xmin, xmax = 0, 0
        ymin, ymax = 0, 0

        if isinstance(ax, matplotlib.axes._subplots.Axes):
            lines = ax.get_lines()

            # if Line2D data does not exist
            if len(lines) == 0:
                XLIM.append(xlim)
                YLIM.append(ylim)

            # if Line2D data does exist
            else:
                for l in lines:
                    xdata = l.get_xdata()
                    ydata = l.get_ydata()
                    xmin, xmax = np.amin(xdata), np.amax(xdata)
                    ymin, ymax = np.amin(ydata), np.amax(ydata)

                    if xmin < xlim[0]:
                        xlim[0] = xmin
                    if xmax > xlim[1]:
                        xlim[1] = xmax
                    if ymin < ylim[0]:
                        ylim[0] = ymin
                    if ymax > ylim[1]:
                        ylim[1] = ymax

                # save limits for each axis object
                if margin != 0:
                    xlim[1] = xlim[1] + margin*xlim[1]
                    ylim[1] = ylim[1] + margin*ylim[1]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                XLIM.append(xlim)
                YLIM.append(ylim)

    if len(XLIM) == 1:
        return (xlim, ylim)
    else:
        return (XLIM, YLIM)

################################################################################
################################################################################
# pickle functions


def pickle_dump(obj, filename='', pickledir='./pickle',
                overwrite=True, verbose=True, **kwargs):
    """
    bug: can't dump figures which used the misc.add_cbar() or misc.add_cbar_ax() function
         because of the use of ax.inset_axes().

    Create pickle directory and pickle.dump object as "<pickledir>/<filename>.pickle"

    Special cases:

      - filename contains relative path: ignores pickledir input
      - filename contains absolute path: ignores pickledir input

    Reason:

        intention is to dump file under the path "filename"

    Args:
        obj (matplotlib.figure.Figure/<any object>)
        filename (str):
          | filename, hardcoded to add ".pickle" extension.
          | Can be relative or absolute path including the filename.
        pickledir (str): default pickle directory
        overwrite (bool): overwrite pickle dumped file if it already exists
        verbose (bool)

    Keyword Args:
        save_as (str): alias of filename

    Returns:
        filepath (str)
            realpath of dumped .pickle file
    """
    if "save_as" in kwargs:
        filename = kwargs["save_as"]
    if filename == "":
        raise TypeError("misc.pickle_dump(): missing 1 required argument: 'filename' or 'save_as'")
    # convention: save files with '.pickle' ending
    if filename[-7:] != ".pickle":
        filename = filename+".pickle"

    filepath = joinpath(filedir=pickledir, filename=filename, create_dir=True)
    if overwrite:
        if os.path.exists(filepath):
            os.remove(filepath)
    with open(filepath, 'wb') as file:
        pl.dump(obj, file)
    if verbose:
        if isinstance(obj, matplotlib.figure.Figure):
            print("pickle.dumped figure as:", os.path.realpath(filepath))
        else:
            print("pickle.dumped object as:", os.path.realpath(filepath))
    return filepath


def _pickle_get_ax_type(fig_or_ax):
    """
    Get axis type by counting <matplotlib.lines.Line2D> and <matplotlib.patches.Rectangle> objects.
    Additionally checks the ratio of Rectangle height:width to detect if barplot was created using
    plt.bar() or plt.barh().

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:
        ax_type (str)
            | "line plot" ~ ax was created using plt.plot()
            | "bar plot"  ~ ax was created using plt.bar()
            | "barh plot" ~ ax was created using plt.barh()
    """
    artists = fig_or_ax.get_default_bbox_extra_artists()

    n_lines = 0
    n_rectangles = 0
    maxh = 0
    maxw = 0

    for item in artists:
        if isinstance(item, matplotlib.lines.Line2D):
            n_lines += 1
        if isinstance(item, matplotlib.patches.Rectangle):
            n_rectangles += 1

            h = item.get_height()
            w = item.get_width()
            if h > maxh:
                maxh = h
            if w > maxw:
                maxw = w

    if (n_rectangles > 10) and (n_rectangles > n_lines):
        if maxh > maxw:
            ax_type = "bar plot"
        if maxw > maxh:
            ax_type = "barh plot"
    else:
        ax_type = "line plot"
    return ax_type


def _pickle_get_ax_data(fig_or_ax):
    """
    Get specific axis data:

      - limits
      - scaling
      - ticks
      - ticklabels

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        fig = fig_or_ax
        ax = fig.axes[0]
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        fig = fig_or_ax.figure
        ax = fig_or_ax

    ### ticklabels are "complicated"
    xticklabels = []
    temp = list(ax.get_xticklabels())
    for item in temp:
        xticklabels.append(item.get_text())

    yticklabels = []
    temp = list(ax.get_yticklabels())
    for item in temp:
        yticklabels.append(item.get_text())

    ax_data = {"ax_type": _pickle_get_ax_type(fig_or_ax),
               "xlim": ax.get_xlim(),
               "ylim": ax.get_ylim(),
               "xscale": ax.get_xscale(),
               "yscale": ax.get_yscale(),
               "xticks": ax.get_xticks(),
               "yticks": ax.get_yticks(),
               "xticklabels": xticklabels,
               "yticklabels": yticklabels}
    return ax_data


__pickle_get_ax_data___append_doc__ = """
        ax_data (dict)
            dictionary containing specific axis data, see table below

            +-------------+--------------+-------------------------------------+
            | key         | value        | description                         |
            +-------------+--------------+-------------------------------------+
            |ax_type (str)| "line plot"  | ax was created using plt.plot()     |
            +             +--------------+-------------------------------------+
            |             | "bar plot"   | ax was created using plt.bar()      |
            +             +--------------+-------------------------------------+
            |             | "barh plot"  | ax was created using plt.barh()     |
            +-------------+--------------+-------------------------------------+
            |xlim/ylim    | tuple        | content of ax.get_xlim()            |
            +-------------+--------------+-------------------------------------+
            |xscale/yscale| str          | content of ax.get_xscale()          |
            +-------------+--------------+-------------------------------------+
            |xticks/yticks| array        | content of ax.get_xticks()          |
            +-------------+--------------+-------------------------------------+
            |x/yticklabels| list         | text content of ax.get_xticklabels()|
            +-------------+--------------+-------------------------------------+
"""


def _pickle_get_line_data(fig_or_ax):
    """
    Get <matplotlib.lines.Line2D> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    artists = fig_or_ax.get_default_bbox_extra_artists()
    line_data = []

    for item in artists:
        if isinstance(item, matplotlib.lines.Line2D):
            data_dict = {}
            data_dict["__description__"] = "line data ~ plt.plot()"
            data_dict["alpha"] = item.get_alpha()
            data_dict["color"] = item.get_color()
            data_dict["label"] = item.get_label()
            data_dict["ls"] = item.get_linestyle()
            data_dict["lw"] = item.get_linewidth()
            data_dict["marker"] = item.get_marker()
            data_dict["mec"] = item.get_mec()
            data_dict["mfc"] = item.get_mfc()
            data_dict["ms"] = item.get_markersize()
            data_dict["xdata"] = item.get_xdata()
            data_dict["ydata"] = item.get_ydata()
            line_data.append(data_dict)
    return line_data


__pickle_get_line_data___append_doc__ = """
        line_data (list)
            list of dicts, which contain data about <matplotlib.lines.Line2D>
            objects with viable Keyword Args as shown in the table below

            +---------+----------------+-------------------------+--------+
            | alpha   | ls (linestyle) | mac (marker edgecolor)  | xdata  |
            +---------+----------------+-------------------------+--------+
            | color   | lw (linewidth) | mfc (marker facecolor)  | ydata  |
            +---------+----------------+-------------------------+--------+
            | label   | marker         | ms (markersize)         |        |
            +---------+----------------+-------------------------+--------+
"""


def _pickle_get_rectangle_data(fig_or_ax):
    """
    Get <matplotlib.patches.Rectangle> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    artists = fig_or_ax.get_default_bbox_extra_artists()
    rect_data = []

    for item in artists:
        if isinstance(item, matplotlib.patches.Rectangle):
            data_dict = {}
            data_dict["__description__"] = "rectangle data ~ plt.bar() or plt.barh()"
            data_dict["alpha"] = item.get_alpha()
            data_dict["bbox"] = item.get_bbox()
            data_dict["bbox_points"] = item.get_bbox().get_points()
            data_dict["ec"] = item.get_ec()
            data_dict["fc"] = item.get_fc()
            data_dict["fill"] = item.get_fill()
            data_dict["height"] = item.get_height()
            data_dict["label"] = item.get_label()
            data_dict["ls"] = item.get_ls()
            data_dict["lw"] = item.get_lw()
            data_dict["width"] = item.get_width()
            data_dict["xy"] = item.get_xy()

            # ignore figure canvas rectangle
            if np.all(data_dict["bbox_points"] == np.array([[0, 0], [1, 1]])):
                pass
            else:
                rect_data.append(data_dict)
    return rect_data


__pickle_get_rectangle_data___append_doc__ = """
        rect_data (list)
            list of dicts, which contain data about <matplotlib.patches.Rectangle>
            objects with viable Keyword Args as shown in the table below

            +----------------+--------+--------+
            | alpha          | fill   | width  |
            +----------------+--------+--------+
            | bbox           | height |  xy    |
            +----------------+--------+--------+
            | bbox_points    | label  |        |
            +----------------+--------+--------+
            | ec (edgecolor) | ls (linestyle)  |
            +----------------+-----------------+
            | fc (facecolor) | lw (linewidth)  |
            +----------------+-----------------+

        .. Hint:: xy are the userland coordinates starting from bottom left
           rectangle corner (rectangle is defined by xy, width, height)
"""
__pickle_get_rectangle_data___bugs_doc__ = """
    Current bugs/problems:
        EDGECOLOR:
        case 1) if plt.bar(..., ec='None', alpha=0.6) was used, then item.get_ec()
                returns array for black color but with alpha value of 0
        case 2) if figure is pickle.dumped and then loaded, the edgecolor values
                are displayed correct (i.e. black with alpha of 0)
        case 3) if plt.bar(...,ec=<read value> , alpha=0.6) is used, all edge
                colors get an alpha value of 0.6

        Although this is somehow expected, but case 1) and case 3) should still
        yield the same results.

        -> workaround is coded in misc.pickle_plot()
    """


def pickle_load(filename, pickledir="./pickle", plot=False):
    """
    pickle.load figure from "<pickledir>/<filename>.pickle". If the pickle file
    contains figure data, auto detects the figure type (i.e. it created using
    plt.plot(), plt.bar() or plt.barh()).

    .. Warning:: %matplotlib notebook backend leaves empty space below figure in
       jupyter notebook when closing figs which were loaded via pickle files.

    Args:
        filename (str): realpath to .pickle file or filename within pickledir
        pickledir (str): default pickle directory
        plot (bool)

    Returns:
        ax_data (dict)
            if pickle file is a figure object
        line_data (list)
            if pickle file is a figure object
        rect_data (list)
            if pickle file is a figure object
        object (?)
            else

    .. Note ::"""
    # NOTE 1: End this docstring with |Notes:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    # test if filename is "fullpath" or "relative path to pickle_dir"
    filepath = joinpath(filedir=pickledir, filename=filename, create_dir=False)
    with open(filepath, "rb") as file:
        obj = pl.load(file)
        if isinstance(obj, matplotlib.figure.Figure):
            fig = obj
            ax_data = _pickle_get_ax_data(fig)
            line_data = _pickle_get_line_data(fig)
            rect_data = _pickle_get_rectangle_data(fig)
            if plot == False:
                fig.set_size_inches(0, 0)
                plt.close()
            return (ax_data, line_data, rect_data)

        else:
            return obj


__pickle_load___append_doc__ = """
    Example:
        | ax_data, line_data, rect_data = pickle_load( < filename > )
        |  # acces data
        | line_data[0]["xdata"]
        | line_data[0]["color"]
"""


def pickle_plot(pickle_files=[], import_settings=True, xscale='auto', yscale='auto',
                align_ylim=True, hist_minorticks=False, **kwargs):
    """
    Creates multifigure from loading pickle_files

    .. Warning:: data is first loaded -> creates a figure -> closed. However in
       jupyter %matplotlib notebook backend leaves empty space below figure when
       closing figs which were loaded via pickle. The empty space is only visible
       in the notebook (i.e. saving a pickle_plot figure is working fine.)

    .. Note: Code for "align" parameter is currently only intended for figures with
        grid = [<any>, 2], i.e. width of 2 figures.

    Args:
        pickle_files (str,list of str): path to pickle file(s) containing the (sub)figure data
        import_settings (bool): apply settings (for line and rectangle objects) from imported pickle_files
        xscale (str):
          | 'auto', 'linear', 'log', 'symlog', 'logit'
          | 'auto': apply xscale of the used pickle_files
        yscale (str):
          | 'auto', 'linear', 'log', 'symlog', 'logit'
          | 'auto': apply yscale of the used pickle_files
        align_ylim (bool): align y axis limits of figures on the same row
        hist_minorticks (bool): turns minorticks (~logscale) for hist figure on/off

    .. Hint:: Args and Keyword Args of misc.figure() are valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"grid": [1, 1]}
    cfg = CONFIG(default, **kwargs)
    if len(pickle_files) != 1 and "grid" not in kwargs:
        nrows = int(round_up(len(pickle_files)/2, base=1))
        ncols = 2
        cfg.grid = [nrows, ncols]

    if isinstance(pickle_files, str):
        pickle_files = [pickle_files]

    elif pickle_files == [] or not isinstance(pickle_files, list):
        exec("help(pickle_plot)")
        raise TypeError("Invalid type of 'pickle_files'. Read the docstring above.")

    fig, ax = figure(**cfg)
    for ndx, pickle in enumerate(pickle_files):
        # skip pickle_files with empty path string
        if pickle == "":
            continue  # python: continue with NEXT loop iteration
        else:
            if len(pickle_files) == 1:
                plt.sca(ax)
            else:
                plt.sca(ax[ndx])
            pass  # python: continue with CURRENT loop iteration aka "do nothing"

        ax_data, line_data, rect_data = pickle_load(pickle)

        ### reconstruct figure
        if ax_data["ax_type"] == "line plot":
            for item in line_data:
                if import_settings:
                    plt.plot(item["xdata"], item["ydata"],
                             alpha=item["alpha"], color=item["color"],
                             label=item["label"], ls=item["ls"], lw=item["lw"],
                             marker=item["marker"], ms=item["ms"],
                             mec=item["mec"], mfc=item["mfc"])
                else:
                    plt.plot(item["xdata"], item["ydata"])

        if ax_data["ax_type"] == "bar plot" or ax_data["ax_type"] == "barh plot":
            for item in rect_data:
                if import_settings:
                    ### workaround start: weird bug with edge colors not properly importing the alpha value
                    if item["ec"] == (0, 0, 0, 0):
                        temp_ec = "None"
                    else:
                        temp_ec = item["ec"]
                    ### workaround end

                    if ax_data["ax_type"] == "bar plot":
                        plt.bar(item["xy"][0], height=item["height"], width=item["width"],
                                fc=item["fc"], ec=temp_ec, alpha=item["alpha"],
                                fill=item["fill"], ls=item["ls"], lw=item["lw"])
                    elif ax_data["ax_type"] == "barh plot":
                        plt.barh(item["xy"][1], height=item["height"], width=item["width"],
                                 fc=item["fc"], ec=temp_ec, alpha=item["alpha"],
                                 fill=item["fill"], ls=item["ls"], lw=item["lw"])

                if not import_settings:
                    if ax_data["ax_type"] == "bar plot":
                        plt.bar(item["xy"][0], height=item["height"], width=item["width"])
                    elif ax_data["ax_type"] == "barh plot":
                        plt.barh(item["xy"][1], height=item["height"], width=item["width"])

        ### general settings
        plt.xlim(ax_data["xlim"])
        plt.ylim(ax_data["ylim"])
        if xscale == "auto":
            plt.xscale(ax_data["xscale"])
        else:
            plt.xscale(xscale)
        if yscale == "auto":
            plt.yscale(ax_data["yscale"])
        else:
            plt.yscale(yscale)

    if align_ylim:
        fig = plt.gcf()
        ax = fig.axes
        for ndx in range(cfg.grid[0]*cfg.grid[1]):
            if ndx % cfg.grid[1] == 0:
                ref_ax = ax[ndx]
            else:
                target_ax = ax[ndx]
            if ndx > 0:
                align_limits(ref_ax=ref_ax, target_ax=target_ax, apply_on='y')
        # """
        # if "grid" in kwargs:
        #     for ndx in range(kwargs["grid"][0]*kwargs["grid"][1]):
        #         i = ndx//kwargs["grid"][1]
        #         j = i + ndx % kwargs["grid"][1]
        #         if i != j:
        #             align_limits(ref_ax=ax[i], target_ax=ax[j], apply_on='y')
        # else:
        #     for i in range(0, len(ax), 2):
        #         align_limits(ref_ax=ax[i], target_ax=ax[i+1], apply_on='y')
        # """
    if not hist_minorticks:
        plt.minorticks_off()

    plt.tight_layout()
    plt.show()
    return (fig, ax)


def hide_figure(fig_or_ax=None, num=None, close=True):
    """
    Alias function of hide_plot(). "Hide" figure by setting its size to (0, 0)
    inches. If number is passed, apply on figure with that number.

    Args:
        fig_or_ax (None, matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        num (None, int): if integer is passed, hide figure with this number
        close (bool): if True, close figure instead of just hiding
    """
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        fig = fig_or_ax
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        fig = fig_or_ax.get_figure()
    elif fig_or_ax is None and num is not None:
        fig = plt.figure(num)

    fig.set_size_inches(0, 0)
    if close and num is None:
        plt.close(fig.number)
    if close and num is not None:
        plt.close(num)

    return


hide_plot = hide_figure


#################################################################################
################################################################################
### limits, ticks and labels functions for plot aesthetics of gridplots

def align_limits(ref_ax, target_ax, apply_on='y', new_lim=[]):
    """
    - Read limits of ref_ax and asign them to target_ax
    - if "new_lim" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y', 'xy'
        new_lim (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_lim == []:
            ax.set_xlim(ax_data["xlim"])
        else:
            ax.set_xlim(new_lim)

    def _apply_on_y(ax):
        if new_lim == []:
            ax.set_ylim(ax_data["ylim"])
        else:
            ax.set_ylim(new_lim)
    ##################################################
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def align_ticks(ref_ax, target_ax, apply_on='y', new_ticks=[]):
    """
    - Read ticks of ref_ax and asign them to target_ax
    - if "new_ticks" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y', 'xy'
        new_ticks (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_ticks == []:
            ax.set_xticks(ax_data["xticks"])
        else:
            ax.set_xticks(new_ticks)

    def _apply_on_y(ax):
        if new_ticks == []:
            ax.set_yticks(ax_data["yticks"])
        else:
            ax.set_yticks(new_ticks)
    ##################################################
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def align_ticklabels(ref_ax, target_ax, apply_on='y', new_ticklabels=[]):
    """
    - Read ticklabels of ref_ax and asign them to target_ax
    - if "new_ticklabels" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y', 'xy'
        new_ticklabels (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_ticklabels == []:
            ax.set_xticklabels(ax_data["xticklabels"])
        else:
            ax.set_xticklabels(new_ticklabels)

    def _apply_on_y(ax):
        if new_ticklabels == []:
            ax.set_yticklabels(ax_data["yticklabels"])
        else:
            ax.set_yticklabels(new_ticklabels)
    ##################################################
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def apply_shared_axes(ax, grid):
    """
    - removes xticklabels of all axes except for bottom row.
    - removes yticklabels of all axes except for left column.

    Args:
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
        grid (list)
            grid of figure (example: grid=[2,2] for 2x2 figure or grid =[3,2] for 3x2 figure)
    """
    ndx = list(range(grid[0]*grid[1]))

    # grid indices of left column
    ndx_left = ndx[::grid[1]]
    # grid indices of bottom row
    ndx_bottom = ndx[grid[0]*grid[1]-grid[1]:]

    for item in [ax[i] for i in ndx if i not in set(ndx_left)]:
        item.set_yticklabels([])
    for item in [ax[i] for i in ndx if i not in set(ndx_bottom)]:
        item.set_xticklabels([])
    return


def convert_ticklabels(axes, multiplier, apply_on='y', prec=0):
    """
    Read ticklabels of axes and multiply with "multiplier"

    Args:
        axes (list of matplotlib.axes._subplots.Axes)
        multiplier (int/float): multiplier for conversion, ie: new_tickvalue = multiplier * old_tickvalue
        apply_on (str): 'x', 'y', 'xy'
        prec (int):
          | precision
          | 0: use integers with no decimal precision
          | n: float with n decimal precision
    """
    for ax in axes:
        if "x" in apply_on:
            # read
            xticklabels = []
            temp = list(ax.get_xticklabels())
            for item in temp:
                xticklabels.append(item.get_text())
            # convert
            if prec == 0:
                xticklabels = [int(float(item)*multiplier) for item in xticklabels if item != ""]
            elif prec > 0:
                xticklabels = [round(float(item)*multiplier, prec) for item in xticklabels if item != ""]
            ax.set_xticklabels(xticklabels)

        if "y" in apply_on:
            # read
            yticklabels = []
            temp = list(ax.get_yticklabels())
            for item in temp:
                yticklabels.append(item.get_text())

            # convert
            if prec == 0:
                yticklabels = [int(float(item)*multiplier) for item in yticklabels if item != ""]
            elif prec > 0:
                yticklabels = [round(float(item)*multiplier, prec) for item in yticklabels if item != ""]
            ax.set_yticklabels(yticklabels)
    return


class number_base_factorization():
    """
    Class to get 10base factorization of any number,
    e.g. 123 = 3*10^0 + 2*10^1 + 1*10^2

    Example:
      | >> x = number_base_factorization(123)
      | >> x() # show content of x
      | self.number: 123
      | self.sign: +
      | self.digits: [1, 2, 3]
      | self.pos_base10_factors: [3, 2, 1]
      | self.neg_base10_factors: None
    """

    def __init__(self, num):
        def get_sign(num):
            """
            signum function
            """
            if num >= 0:
                return "+"
            else:
                return "-"

        def get_digits(num):
            """
            Convert <num> into list of digits.
            """
            digits = [int(d) if d not in set("-.") else d for d in str(num)]
            if "-" in digits:
                del digits[digits.index("-")]
            return digits

        def base10_factorize(digits):
            """
            get base10 factorization of a list with digits.

            Args:
                digits (list)

            Returns:
                pos_base10_factors (list)
                    list with positive factors
                neg_base10_factors (list)
                    list with negative factors
            """
            if "." in digits:
                temp = digits.index(".")
                pos_base10_factors = digits[:temp][::-1]
                neg_base10_factors = digits[temp+1:]
            else:
                pos_base10_factors = digits[::-1]
                neg_base10_factors = None
            return pos_base10_factors, neg_base10_factors

        self.number = num
        self.sign = get_sign(num)
        self.digits = get_digits(num)
        temp = base10_factorize(self.digits)
        self.pos_base10_factors = temp[0]
        self.neg_base10_factors = temp[1]

    def __call__(self):
        """
        Print class content.
        """
        print("self.number:", self.number)
        print("self.sign:", self.sign)
        print("self.digits:", self.digits)
        print("self.pos_base10_factors:", self.pos_base10_factors)
        print("self.neg_base10_factors:", self.neg_base10_factors)
        return


def setup_ticks(vmin, vmax, major_base, minor_base=None):
    """
    Setup axis ticks of a plot in vmin <= ticks <= vmax.

    Args:
        vmin (int, float)
        vmax (int, float)
        major_base (int, float)
        minor_base (int, float)

    Returns:
        majorticks (list)
            list with majorticks
        minorticks (list)
            list with minorticks
    """
    min_ = round_down(vmin, base=major_base)
    max_ = round_up(vmax, base=major_base)

    majorticks = np.arange(min_, max_+major_base, major_base)
    majorticks = [i for i in majorticks if i >= vmin if i <= vmax]
    if minor_base is not None:
        minorticks = np.arange(min_+minor_base, max_, minor_base)
        minorticks = [i for i in minorticks if i not in majorticks]
    else:
        minorticks = list()
    return majorticks, minorticks


def setup_logscale_ticks(vmax):
    """
    Setup majorticks and minorticks for logscale with ticks <= vmax.

    Args:
        vmax (int, float)

    Returns:
        majorticks (list)
            ticks at 1, 10, 100, 1000, etc.
        minorticks (list):
            ticks at 2, 3, 4,..., 8, 9, 20,30,40,...,80,90, etc.
    """
    vmax_factorized = number_base_factorization(vmax)
    majorticks = [[1]]
    minorticks = []

    for ndx, item in enumerate(vmax_factorized.pos_base10_factors):
        major, minor = setup_ticks(vmin=10**ndx, vmax=10**(ndx+1), major_base=10**(ndx+1), minor_base=10**ndx)
        majorticks.append(major)
        minorticks.append(minor)

    majorticks = flatten_array(majorticks)
    minorticks = flatten_array(minorticks)
    return majorticks, minorticks


def set_logscale_ticks(ax, apply_on="x", vmax=None, minorticks=True, **kwargs):
    """
    Apply logscale ticks on ax. If vmax is specified, sets vmax as upper limit.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        apply_on (str): "x", "y", "xy"
        vmin (None, int, float): highest logscale tick value
        minorticks (bool): display minorticks on/off

    Keyword Args:
        xmin (None, int, float)
        xmax (None, int, float)
        ymin (None, int, float)
        ymax (None, int, float)
    """
    default = {"vmax": vmax,
               "xmin": None,
               "xmax": None,
               "ymin": None,
               "ymax": None}
    cfg = CONFIG(default, **kwargs)

    if "x" in apply_on:
        if cfg.vmax is None:
            _, vmax = ax.get_xlim()
        major, minor = setup_logscale_ticks(vmax)
        ax.set_xscale("log")
        if cfg.xmin is None:
            cfg.xmin, _ = ax.get_xlim()
        if cfg.xmax is None:
            _, cfg.xmax = ax.get_xlim()
        ax.set_xticks([], minor=True)
        ax.set_xticks(major)
        if minorticks:
            ax.set_xticks(minor, minor=True)

        # set upper limit via vmax (not cfg.vmax)
        if vmax is None:
            ax.set_xlim(cfg.xmin, cfg.xmax)
        else:
            ax.set_xlim(cfg.xmin, vmax)

    if "y" in apply_on:
        if cfg.vmax is None:
            _, vmax = ax.get_ylim()
        major, minor = setup_logscale_ticks(vmax)
        ax.set_yscale("log")
        if cfg.ymin is None:
            cfg.ymin, _ = ax.get_ylim()
        if cfg.ymax is None:
            _, cfg.ymax = ax.get_ylim()
        ax.set_yticks([], minor=True)
        ax.set_yticks(major)
        if minorticks:
            ax.set_yticks(minor, minor=True)

        # set upper limit via vmax (not cfg.vmax)
        if vmax is None:
            ax.set_ylim(cfg.ymin, cfg.ymax)
        else:
            ax.set_ylim(cfg.ymin, vmax)

    return


#################################################################################
################################################################################
### colorbar functions


def create_cmap(seq, vmin=None, vmax=None, ax=None):
    """
    Return a LinearSegmentedColormap

    Args:
        seq (sequence): sequence of color strings and floats. The floats describe the color thresholds and
           should be increasing and in the interval (0,1).
        vmin (float): min value of cmap
        vmax (float): max value of cmap
        ax (None, ax): figure ax for adding colorbar

    Returns:
        cmap (LinearSegmentedColormap)

    Example:
        seq = ["lightblue", 2/6, "lightgreen", 3/6, "yellow", 4/6, "orange", 5/6, "red"]
        cmap = misc.create_cmap(seq, vmin=0, vmax=12)
    """
    for ndx, item in enumerate(seq):
        if isinstance(item, str):
            seq[ndx] = matplotlib.colors.ColorConverter().to_rgb(item)
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    cmap = matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)

    # add colorbar if ax is passed
    if ax is not None:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 1
        _norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=_norm, cmap=cmap), ax=ax)
    return cmap


def add_cbar_ax(ax, bounds="auto", location="right", orientation="vertical"):
    """
    Add colorbar ax relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        bounds (str,list):
          | str: "auto": apply bounds based on kws 'orientation' and 'location'.
          | list: [x0, y0, height, width] using axes coordinates.
          |       Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Returns:
        cbar_ax (matplotlib.axes._axes.Axes)
            colorbar ax
    """
    # apply passed bounds and ignore kws 'orientation' and 'location'
    if bounds != "auto":
        pass

    # apply bounds based on kws 'orientation' and 'location'
    else:
        if location == "right":
            x0, y0 = 1.05, 0
        elif location == "left":
            x0, y0 = -0.1, 0
        elif location == "top":
            x0, y0 = 0, 1.1
        elif location == "bottom":
            x0, y0 = 0, -0.2

        if orientation == "horizontal":
            h, w = 0.05, 1
        elif orientation == "vertical":
            h, w = 1, 0.05

        bounds = [x0, y0, w, h]

    cbar_ax = ax.inset_axes(bounds)
    return cbar_ax


def add_cbar(ax, cbar_ax=None, cmap=None, bounds="auto", location="right", orientation="vertical", **kwargs):
    """
    Draw colorbar relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        cbar_ax (None/matplotlib.axes._axes.Axes):
          | colorbar ax
          | None: add new cbar_ax relative to existing ax
          | matplotlib.axes._axes.Axes: use passed cbar_ax (can be created using misc.add_cbar_ax()).
        cmap (None, LinearSegmentedColormap): output of create_cmap()
        bounds (str, list):
          | str: "auto": apply bounds based on kws 'orientation' and 'location'.
          | list: [x0, y0, height, width] using axes coordinates.
          |       Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Keyword Args:
        vmin (None, float): colorbar min value
        vmax (None, float): colorbar max value
        cbar_label/label (None, str)
        cbar_fontweight/fontweight (None, str): "bold", "normal"
        cbar_location/location (None, str): "right", "bottom", "left", "top"
        cbar_orientation/orientation (None, str): "vertical", "horizontal"

    Returns:
        cbar (matplotlib.colorbar.Colorbar)
            color bar
    """
    default = {"vmin": None,
               "vmax": None,
               "cbar_label": None,
               "cbar_fontweight": "bold",
               "cbar_location": location,
               "cbar_orientation": orientation
               }

    cfg = CONFIG(default, **kwargs)
    cfg.update_by_alias(alias="label", key="cbar_label", **kwargs)
    cfg.update_by_alias(alias="fontweight", key="cbar_fontweight", **kwargs)

    if cbar_ax is None:
        cbar_ax = add_cbar_ax(ax, bounds,
                              location=cfg.cbar_location,
                              orientation=cfg.cbar_orientation)

    for item in ax.get_children():
        if isinstance(item, matplotlib.collections.QuadMesh):
            QuadMesh = item  # QuadMesh ~ sns.heatmap() plots

    if "QuadMesh" in locals():
        # mappable must be matplotlib.cm.ScalarMappable ~ sns.heatmap()
        # see help(fig.colorbar) for more info
        cbar = plt.colorbar(mappable=QuadMesh, cax=cbar_ax, orientation=cfg.cbar_orientation)
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        if cfg.vmin is None:
            cfg.vmin = 0
        if cfg.vmax is None:
            cfg.vmax = 1
        _norm = matplotlib.colors.Normalize(vmin=cfg.vmin, vmax=cfg.vmax)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=_norm, cmap=cmap), cax=cbar_ax, orientation=cfg.cbar_orientation)
    else:
        #print("misc.add_cbar() works currently only with sns.heatmap() plots.")
        cbar = plt.colorbar(cax=cbar_ax, orientation=cfg.cbar_orientation)

    cbar.set_label(label=cfg.cbar_label, weight=cfg.cbar_fontweight)
    cbar_set_ticks_position(cbar, location=cfg.cbar_location)
    return cbar


def cbar_set_ticks_position(cbar, location):
    """
    Set ticks position of colorbar.

    Args:
        cbar (matplotlib.colorbar.Colorbar)
        location (str): "right", "bottom", "left", "top"
    """
    if location == "top" or location == "bottom":
        cbar.ax.xaxis.set_ticks_position(location)
    if location == "right" or location == "left":
        cbar.ax.yaxis.set_ticks_position(location)
    return


#################################################################################
################################################################################
### complete / append docstrings


def INIT_append_docstrings():
    """
    Some functions in this module have split up docstrings.
    The intention of this design choice is to assure that docstrings which rely
    on each other are matching and always up to date. This function is executed
    ONCE upon initialization of the module and appends the missing parts.
    """
    _pickle_get_ax_data.__doc__ += __pickle_get_ax_data___append_doc__
    _pickle_get_line_data.__doc__ += __pickle_get_line_data___append_doc__
    _pickle_get_rectangle_data.__doc__ += __pickle_get_rectangle_data___append_doc__ + __pickle_get_rectangle_data___bugs_doc__

    for item in [__pickle_get_ax_data___append_doc__,
                 __pickle_get_line_data___append_doc__,
                 __pickle_get_rectangle_data___append_doc__,
                 __pickle_load___append_doc__]:
        pickle_load.__doc__ += item
    return


INIT_append_docstrings()
