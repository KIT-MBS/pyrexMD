import myPKG.core as core
#from myPKG.core import iColor, iWidgets, iPlayer, iColor
import myPKG.analysis as ana
import myPKG.misc as misc


version = 1.0


def init():
    """
    Init important modules of myPKG and other packages
    by copy pasting the printed message.
    """
    print("""
copy&paste in jupyter notebook:

%matplotlib notebook

from IPython.display import display
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
import myPKG.core as core
import myPKG.misc as misc
import myPKG.analysis as ana
import myPKG.gmx as gmx
    """)
    return
