# @Author: Arthur Voronin
# @Date:   06.05.2021
# @Filename: dihedrals.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

"""
.. hint:: This module contains functions related to dihedral analyses.

Example:
--------

.. code-block:: python

    import MDAnalysis as mda
    import pyrexMD.analysis.dihedrals as dih

    mobile = mda.Universe("path/to/tpr", "path/to/xtc")

    # get dihedral angles of current frame
    phi = dih.get_phi_values(u, sel="protein and resid 1-3")
    psi = dih.get_psi_values(u, sel="protein and resid 1-3")
    omega = dih.get_omega_values(u, sel="protein and resid 1-3")

    # plot ramachandran
    rama = dih.get_ramachandran(mobile)

Content:
--------
"""


import pyrexMD.misc as _misc
import pyrexMD.topology as _top
from MDAnalysis.analysis import dihedrals
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Cannot determine phi and psi angles for the first")
warnings.filterwarnings("ignore", message="All ALA, CYS, GLY, PRO, SER, THR, and VAL residues")


def get_ramachandran(u, sel="protein", sss=[None, None, None], plot=True, **kwargs):
    """
    Get Ramachandran information (phi & psi dihedral values, frames, etc).

    .. Note:: phi/psi angles are not defined for first and last residues.

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        plot (bool)

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        color (str): "black"
        marker (str): "."
        ref (bool): True: Adds a general Ramachandran plot which shows allowed
          (dark blue ~ 90% data points) and marginally allowed regions (light
          blue ~ 99% data points). Lookup MDAnalysis online documentation about
          dihedrals for more information.
        num (None, int):
          | figure number
          | None: create new figure
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        rama (MDAnalysis.analysis.dihedrals.Ramachandran)
            class contains atom groups, angles, frames, times, etc
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "color": "black",
               "marker": ".",
               "ref": True,
               "num": None,
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if "ref" in kwargs:
        del kwargs["ref"]
    if "num" in kwargs:
        del kwargs["num"]
    if cfg.norm:
        _top.norm_universe(u)
    ############################################################################
    rama = dihedrals.Ramachandran(u.select_atoms(sel)).run(start=cfg.start, stop=cfg.stop, step=cfg.step)

    _misc.figure(num=cfg.num, **kwargs)
    rama.plot(color=cfg.color, marker=cfg.marker, ref=cfg.ref)
    plt.tight_layout()
    return rama


def get_janin(u, sel="protein", sss=[None, None, None], plot=True, verbose=True, **kwargs):
    """
    Get Janin information (chi1 & chi2 dihedral values, frames, etc).

    .. Note:: chi1/chi2 angles are not defined for ALA, CYS, GLY, PRO, SER, THR, VAL

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        plot (bool)
        verbose (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                        residues have been removed from the selection."

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        color (str): "black"
        marker (str): "."
        ref (bool): True: adds a general Janin plot which shows allowed (dark
          blue ~ 90% data points) and marginally allowed regions (light blue ~
          99% data points). Lookup MDAnalysis online documentation about
          dihedrals for more information.
        num (None, int):
          | figure number
          | None: create new figure
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        janin (MDAnalysis.analysis.dihedrals.Janin)
            class contains atom groups, angles, frames, times, etc
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "color": "black",
               "marker": ".",
               "ref": True,
               "num": None,
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if "ref" in kwargs:
        del kwargs["ref"]
    if "num" in kwargs:
        del kwargs["num"]
    if cfg.norm:
        _top.norm_universe(u)
    ############################################################################
    janin = dihedrals.Janin(u.select_atoms(sel)).run(start=cfg.start, stop=cfg.stop, step=cfg.step)

    _misc.figure(num=cfg.num, **kwargs)
    janin.plot(color=cfg.color, marker=cfg.marker, ref=cfg.ref)
    plt.tight_layout()

    if verbose:
        _misc.cprint("All ALA, CYS, GLY, PRO, SER, THR, and VAL residues have been removed from the selection.", "red")
    return janin


def get_phi_values(u, sel="protein", sss=[None, None, None], **kwargs):
    """
    Get phi dihedral values.

    .. Note:: phi/psi angles are not defined for first and last residues.

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
         | [start, stop, step]
         | start (None, int): start frame
         | stop (None, int): stop frame
         | step (None, int): step size

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        phi (array)
            array with phi dihedral values
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.norm:
        _top.norm_universe(u)
    ############################################################################
    a = [res.phi_selection() for res in u.select_atoms(sel).residues
         if res.phi_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    phi_values = dih.angles
    return phi_values


def get_psi_values(u, sel="protein", sss=[None, None, None], **kwargs):
    """
    Get psi dihedral values.

    .. Note:: phi/psi angles are not defined for first and last residues.

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
         | [start, stop, step]
         | start (None, int): start frame
         | stop (None, int): stop frame
         | step (None, int): step size

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        psi_values (array)
            array with psi dihedral values
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.norm:
        _top.norm_universe(u)
    ############################################################################
    a = [res.psi_selection() for res in u.select_atoms(sel).residues
         if res.psi_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    psi_values = dih.angles
    return psi_values


def get_omega_values(u, sel="protein", sss=[None, None, None], **kwargs):
    """
    Get omega dihedral values.

    Args:
        u (Muniverse)
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        omega_values (array)
            array with omega dihedral values
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.norm:
        _top.norm_universe(u)
    ############################################################################
    a = [res.omega_selection() for res in u.select_atoms(sel).residues
         if res.omega_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    omega_values = dih.angles
    return omega_values


def get_chi1_values(u, sel="protein", sss=[None, None, None], warn=True, **kwargs):
    """
    Get chi1 dihedral values.

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        warn (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                     residues have been removed from the selection."

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        chi1_values (array)
            array with chi1 dihedral values
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.norm:
        _top.norm_universe(u)
    if warn:
        _misc.cprint("All ALA, CYS, GLY, PRO, SER, THR, and VAL residues have been removed from the selection.", "red")
    ############################################################################
    a = [res.chi1_selection() for res in u.select_atoms(sel).residues
         if res.chi1_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    chi1_values = dih.angles
    return chi1_values


def get_chi2_values(u, sel="protein", sss=[None, None, None], warn=True, **kwargs):
    """
    Get chi2 dihedral values.

    Args:
        u (universe)
        sel (str): selection string
        sss (list):
          | [start, stop, step]
          | start (None, int): start frame
          | stop (None, int): stop frame
          | step (None, int): step size
        warn (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                     residues have been removed from the selection."

    Keyword Args:
        start (None, int): start frame
        stop (None, int): stop frame
        step (None, int): step size
        norm (bool): norm universe before calculation. Defaults to True.

    Returns:
        chi2_values (array)
            array with chi2 dihedral values
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "norm": True}
    cfg = _misc.CONFIG(default, **kwargs)
    if cfg.norm:
        _top.norm_universe(u)
    if warn:
        _misc.cprint("All ALA, CYS, GLY, PRO, SER, THR, and VAL residues have been removed from the selection.", "red")
    ############################################################################
    try:
        a = [res.chi2_selection() for res in u.select_atoms(sel).residues
             if res.chi2_selection() is not None]
    except AttributeError:
        raise AttributeError("Selection has no chi2 angles.")
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    chi2_values = dih.angles
    return chi2_values
