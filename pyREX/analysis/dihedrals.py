# @Author: Arthur Voronin <arthur>
# @Date:   06.05.2021
# @Filename: dihedrals.py
# @Last modified by:   arthur
# @Last modified time: 08.05.2021


import pyREX.misc as _misc
from MDAnalysis.analysis import dihedrals
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Cannot determine phi and psi angles for the first")
warnings.filterwarnings("ignore", message="All ALA, CYS, GLY, PRO, SER, THR, and VAL residues")


def get_ramachandran(u, sel="protein", sss=[None, None, None], plot=True, **kwargs):
    """
    get Ramachandran information (phi & psi dihedral values, n_frames, etc).

    Note:
        phi/psi angles are not defined for first and last residues.

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        plot (bool)

    Kwargs:
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        color (str): "black"
        marker (str): "."
        ref (bool):
            True: Adds a general Ramachandran plot which shows allowed (dark
                  blue ~ 90% data points) and marginally allowed regions (light
                  blue ~ 99% data points).
                  -> lookup MDAnalysis online documentation about dihedrals.
        num (None/int): figure number
            None: create new figure

    Returns:
        rama (MDAnalysis.analysis.dihedrals.Ramachandran): class contains, e.g.
            - atom groups
            - angles
            - frames
            - times
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "color": "black",
               "marker": ".",
               "ref": True,
               "num": None}
    cfg = _misc.CONFIG(default, **kwargs)
    if "ref" in kwargs:
        del kwargs["ref"]
    if "num" in kwargs:
        del kwargs["num"]
    ############################################################################
    rama = dihedrals.Ramachandran(u.select_atoms(sel)).run(start=cfg.start, stop=cfg.stop, step=cfg.step)

    _misc.figure(num=cfg.num, **kwargs)
    rama.plot(color=cfg.color, marker=cfg.marker, ref=cfg.ref)
    plt.tight_layout()
    return rama


def get_janin(u, sel="protein", sss=[None, None, None], plot=True, verbose=True, **kwargs):
    """
    get Janin information (chi1 & chi2 dihedral values, n_frames, etc).

    Note:
        chi1/chi2 angles are not defined for ALA, CYS, GLY, PRO, SER, THR, VAL

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        plot (bool)
        verbose (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                        residues have been removed from the selection."

    Kwargs:
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size
        color (str): "black"
        marker (str): "."
        ref (bool):
            True: Adds a general Janin plot which shows allowed (dark
                  blue ~ 90% data points) and marginally allowed regions (light
                  blue ~ 99% data points).
                  -> lookup MDAnalysis online documentation about dihedrals.
        num (None/int): figure number
            None: create new figure

    Returns:
        janin (MDAnalysis.analysis.dihedrals.Janin): class contains, e.g.
            - atom groups
            - angles
            - frames
            - times
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2],
               "color": "black",
               "marker": ".",
               "ref": True,
               "num": None}
    cfg = _misc.CONFIG(default, **kwargs)
    if "ref" in kwargs:
        del kwargs["ref"]
    if "num" in kwargs:
        del kwargs["num"]
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

    Note:
        phi/psi angles are not defined for first and last residues.

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size

    Returns:
        phi (array)
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    a = [res.phi_selection() for res in u.select_atoms(sel).residues
         if res.phi_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    phi_values = dih.angles
    return phi_values


def get_psi_values(u, sel="protein", sss=[None, None, None], **kwargs):
    """
    Get psi dihedral values.

    Note:
        phi/psi angles are not defined for first and last residues.

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size

    Returns:
        psi_values (array)
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)
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
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size

    Returns:
        omega_values (array)
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    a = [res.omega_selection() for res in u.select_atoms(sel).residues
         if res.omega_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    omega_values = dih.angles
    return omega_values


def get_chi1_values(u, sel="protein", sss=[None, None, None], verbose=True, **kwargs):
    """
    Get chi1 dihedral values.

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        verbose (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                        residues have been removed from the selection."

    Kwargs:
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size

    Returns:
        chi1_values (array)
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    a = [res.chi1_selection() for res in u.select_atoms(sel).residues
         if res.chi1_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    chi1_values = dih.angles

    if verbose:
        _misc.cprint("All ALA, CYS, GLY, PRO, SER, THR, and VAL residues have been removed from the selection.", "red")
    return chi1_values


def get_chi2_values(u, sel="protein", sss=[None, None, None], verbose=True, **kwargs):
    """
    Get chi2 dihedral values.

    Args:
        u (MDA universe)
        sel (str): selection string
        sss (list): [start, stop, step]
            start (None/int): start frame
            stop (None/int): stop frame
            step (None/int): step size
        verbose (bool): print warning "All ALA, CYS, GLY, PRO, SER, THR, and VAL
                        residues have been removed from the selection."

    Kwargs:
        start (None/int): start frame
        stop (None/int): stop frame
        step (None/int): step size

    Returns:
        chi2_values (array)
    """
    default = {"start": sss[0],
               "stop": sss[1],
               "step": sss[2]}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    a = [res.chi2_selection() for res in u.select_atoms(sel).residues
         if res.chi2_selection() is not None]
    dih = dihedrals.Dihedral(a).run(start=cfg.start, stop=cfg.stop, step=cfg.step)
    chi2_values = dih.angles

    if verbose:
        _misc.cprint("All ALA, CYS, GLY, PRO, SER, THR, and VAL residues have been removed from the selection.", "red")
    return chi2_values
