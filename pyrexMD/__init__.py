# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: __init__.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

import pyrexMD
import pyrexMD.core as core
import pyrexMD.gmx as gmx
import pyrexMD.misc as misc
import pyrexMD.topology as topology
import pyrexMD.analysis as analysis
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Element information is absent or missing for a few")
warnings.filterwarnings("ignore", message="Failed to guess the mass for the following atom types")
warnings.filterwarnings("ignore", message="Unit cell dimensions not found.")
warnings.filterwarnings("ignore", message="1 A\^3 CRYST1 record,")
warnings.filterwarnings("ignore", message="Found no information for attr:")
warnings.filterwarnings("ignore", message="!!! WARNING !!! Manipulating the local contents of a DNDarray needs")
# MDAnalysis throws many unnecessary warnings when warnings module is used

plt.style.use(f"{pyrexMD.__path__[0]}/.mplstyle")


__version__ = 1.0


def init():
    """
    Init modules of pyrexMD and other important packages into new jupyter notebook
    session by copy&pasting the printed message.
    """
    print("""
copy&paste in jupyter notebook:

%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
import pyrexMD.core as core
import pyrexMD.gmx as gmx
import pyrexMD.misc as misc
import pyrexMD.topology as top
import pyrexMD.analysis.analysis as ana
#import pyrexMD.analysis.contacts as con
#import pyrexMD.analysis.cluster as clu
#import pyrexMD.analysis.dihedrals as dih
#import pyrexMD.analysis.gdt as gdt
#import pyrexMD.rex as rex

from tqdm.notebook import tqdm
    """)
    return
