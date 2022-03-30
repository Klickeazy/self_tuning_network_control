import numpy as np
import networkx as netx
from copy import deepcopy as dc
import pickle

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['legend.title_fontsize'] = 10
matplotlib.rcParams['legend.framealpha'] = 0.5
matplotlib.rcParams['lines.markersize'] = 5
# matplotlib.rcParams['image.cmap'] = 'Blues'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


def system_package(A_in=None, B_in=None, X0_in=None, W_in=None, Q_in=None, R_in=None, system_label=None):

    A = dc(A_in)
    nx = np.shape(A)[0]

    B = dc(B_in)
    if B is None:
        B = np.zeros_like(A)
    nu = np.shape(B)[1]

    X0 = dc(X0_in)
    if X0 is None:
        X0 = np.zeros(nx)
        metric1 = 0
        # no initial state => eigmax(P0)
    else:
        metric1 = np.ndim(X0)
        # metric = 1 => initial state vector => x0 * P0 * x0
        # metric = 2 => covariance of zero-mean inital state distribution => trace(P0 * X0)

    W = dc(W_in)
    # covariance of zero-mean additive disturbance - set to matrix of 0s for no additive disturbance

    Q = dc(Q_in)
    if Q is None:
        Q = np.identity(nx)
    R = dc(R_in)
    if R is None:
        R = np.identity(nu)

    system_name = dc(system_label)

    system = {'A': A, 'B': B, 'X0': X0, 'W': W, 'Q': Q, 'R': R, 'name': system_name, 'cost_metric': metric1}
    return_values = {'System': system}
    return return_values

#####################################


def solve_discreteLQR(sys_in=None):

