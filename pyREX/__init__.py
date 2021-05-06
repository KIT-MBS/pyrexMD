import pyREX.misc as misc
import pyREX.core as core
import pyREX.analysis as ana


version = 1.0


def init():
    """
    Init important modules of pyREX and other packages
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
import pyREX.misc as misc
import pyREX.core as core
import pyREX.analysis as ana
#import pyREX.abinitio as abi
import pyREX.cluster as cluster
import pyREX.gmx as gmx
misc.apply_matplotlib_rc_settings()
    """)
    return
