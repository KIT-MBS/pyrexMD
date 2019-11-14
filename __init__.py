from myPKG.core import iColor, iWidgets, iPlayer, iColor
import myPKG.analysis
import myPKG.rex
import myPKG.misc

version = 1.0


def init():
    """
    Init all modules of myPKG and other important packages
    by copy pasting the printed message.
    """
    print("""
copy&paste in jupyter notebook:

%matplotlib notebook

from IPython.display import display
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import nglview as ngl
import myPKG.core as core
import myPKG.analysis as ana
import myPKG.rex as rex
import myPKG.misc as misc
    """)
    return
