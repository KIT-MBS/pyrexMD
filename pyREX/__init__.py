# @Author: Arthur Voronin <arthur>
# @Date:   17.04.2021
# @Filename: __init__.py
# @Last modified by:   arthur
# @Last modified time: 11.05.2021


import pyREX.misc as misc
import pyREX.core as core
import pyREX.analysis as analysis
import warnings
# MDAnalysis throws many unnecessary warnings when warnings module is used
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Element information is absent or missing for a few")
warnings.filterwarnings("ignore", message="Failed to guess the mass for the following atom types")
warnings.filterwarnings("ignore", message="Unit cell dimensions not found.")
warnings.filterwarnings("ignore", message="1 A\^3 CRYST1 record,")
warnings.filterwarnings("ignore", message="Found no information for attr:")


def Universe(top, traj=None, tpr_resid_from_one=True, **kwargs):
    """
    returns mda.Universe(top, traj, tpr_resid_from_one=tpr_resid_from_one, **kwargs)
    """
    return mda.Universe(top, traj, tpr_resid_from_one=tpr_resid_from_one, **kwargs)


def init():
    """
    Init important modules of pyREX and other packages
    by copy pasting the printed message.
    """
    print("""
copy&paste in jupyter notebook:

%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
import pyREX.misc as misc
import pyREX.core as core
import pyREX.analysis.analysis as ana
#import pyREX.analysis.contacts as con
#import pyREX.analysis.dihedrals as dih
#import pyREX.analysis.GDT as gdt
#import pyREX.decoy.abinitio as abi
#import pyREX.decoy.cluster as clu
#import pyREX.gmx as gmx
#import pyREX.rex as rex

from tqdm.notebook import tqdm
misc.apply_matplotlib_rc_settings()
    """)
    return
