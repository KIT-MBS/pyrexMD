import myPKG.misc as misc
import myPKG.core as core
import myPKG.analysis as ana


version = 1.0


def init():
    """
    Init important modules of myPKG and other packages
    by copy pasting the printed message.
    """
    print("""
copy&paste in jupyter notebook:

%matplotlib notebook

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
import myPKG.misc as misc
import myPKG.core as core
import myPKG.analysis as ana
import myPKG.abinitio as abi
import myPKG.cluster as cluster
import myPKG.gmx as gmx
misc.apply_matplotlib_rc_settings()
    """)
    return
