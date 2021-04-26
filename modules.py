#!/usr/bin/env python

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.linalg import expm as mexp

from pylab import *
import matplotlib.style
import matplotlib as mpl
from matplotlib.colors import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import fontManager, FontProperties

mpl.rcParams['text.usetex'] = True
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.markersize'] = 6.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.sans-serif'] = 'Computer Modern Roman'
ratio=(np.sqrt(5)-1)/2
plt.rcParams["figure.figsize"] = 3.37, 3.37*ratio
col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'k']
# Uncomment line below to search rcParams for useful key.
#print(rcParams.keys())

