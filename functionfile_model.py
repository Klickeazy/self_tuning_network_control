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


def system_package1(A_in=None, B_in=None, X0_in=None, W_in=None, Q_in=None, R_in=None, system_label=None):

    A = dc(A_in)
    nx = np.shape(A)[0]

    B = dc(B_in)
    if B is None:
        B = np.zeros_like(A)
    nu = np.shape(B)[1]

    X0 = dc(X0_in)
    if X0 is None:
        X0 = np.zeros(nx)
        metric_fn = metric0
        # no initial state => eigmax(P0)
    elif np.ndim(X0) == 1:
        metric_fn = metric1
        # metric = 1 => initial state vector => x0 * P0 * x0
    elif np.ndim(X0) == 2:
        metric_fn = metric2
        # metric = 2 => covariance of zero-mean inital state distribution => trace(P0 * X0)
    else:
        raise Exception('Check initial conditions X0 - affects cost metrics function')

    W = dc(W_in)
    # covariance of zero-mean additive disturbance - set to matrix of 0s for no additive disturbance

    Q = dc(Q_in)
    if Q is None:
        Q = np.identity(nx)
    R = dc(R_in)
    if R is None:
        R = np.identity(nu)

    system_name = dc(system_label)

    system = {'A': A, 'B': B, 'X0': X0, 'W': W, 'Q': Q, 'R': R, 'name': system_name, 'cost_metric': metric_fn, 'sys_type': 1}
    return_values = {'System': system}
    return return_values


#####################################


def metric0(P_in, sys_in):
    # Metric for no initial state data => worst case cost over unit circle distribution of initial states
    J = np.max(np.linalg.eigvals(P_in))

    return_vals = {'J': J}
    return return_vals


def metric1(P_in, sys_in):
    # Metric for initial state vector => cost for given initial state vector
    J = sys_in['X0'].T @ P_in @ sys_in['X0']

    return_vals = {'J': J}
    return return_vals


def metric2(P_in, sys_in):
    # Metric for covariance of zero-mean initial state distribution => trace of cost scaled with covariance
    J = np.trace(P_in @ sys_in['X0'])

    return_vals = {'J': J}
    return return_vals


#####################################


def solve_constraints_initializer():

    # Maximum cost metric bound - set None to look for convergence rather than
    J_max = 10**8

    # Minimum convergence accuracy for cost matrix over iterations
    P_accuracy = 10**(-4)

    # Time horizon of simulation/recursion
    T_max = 200

    return_values = {'J_max': J_max, 'T_max': T_max, 'P_accuracy': P_accuracy}
    return return_values


#####################################


def matrix_convergence_check(A, B, accuracy, check_type=None):
    np_norm_methods = ['inf', 'fro', 2, None]
    if check_type not in np_norm_methods:
        raise Exception('Check convergence method')
    if check_type is None:
        return np.allclose(A, B, a_tol=accuracy)
    elif check_type in np_norm_methods and accuracy is not None:
        return np.norm(A-B, ord=check_type) < accuracy


#####################################


def recursion_lqr_1step(P_t1, Sys_in):
    A = Sys_in['A']
    B = Sys_in['B']

    Q = Sys_in['Q']
    R = Sys_in['R']

    K_t0 = -np.linalg.inv(R + B.T@P_t1@B)@B.T@P_t1@A
    P_t0 = Q + A.T@P_t1@A + A.T@P_t1@B@K_t0

    return_vals = {'P': P_t0, 'K': K_t0}
    return return_vals


#####################################


def solve_discreteLQR(sys_in=None, solve_constraints_in=None, cost_fn=recursion_lqr_1step):

    if solve_constraints_in is None:
        solve_constraints = solve_constraints_initializer()
    else:
        solve_constraints = dc(solve_constraints_in)


    Sys = dc(sys_in)

    A = Sys['A']
    B = Sys['B']

    Q = Sys['Q']
    R = Sys['R']

    if np.shape(A)[1] != np.shape(Q)[0]:
        raise Exception('Check A vs Q dimensions - number of states do not match')
    if np.shape(B)[1] != np.shape(R)[0]:
        raise Exception('ERROR: check B vs R dimensions - number of inputs')

    P = dc(Q)
    P_check = 0

    t_steps = 0
    while not P_check:  # P_check = 0 for recursion to continue, 1 for successful exit, 2 for failure exit

        try:
            rec_calc = cost_fn(P, Sys)
        except:
            raise Exception('Cost calculation issue')

        t_steps += 1

        P_t = rec_calc['P']
        if solve_constraints['J_max'] is not None and Sys['cost_metric'](P_t, Sys) > solve_constraints['J_max']:
            # Exceeding design cost bounds
            P_check = 2
            break

        if solve_constraints['J_max'] is None and matrix_convergence_check(P_t, P, solve_constraints['P_accuracy'], None):
            # Convergence of cost function
            P_check = 1
            break

        if solve_constraints['T_max'] < t_steps:
            # Exceeded computation steps - success for finite horizon, fail for infinite horizon
            if solve_constraints['J_max'] is not None:  # Success for bounded costs over finite horizon
                P_check = 1
                break
            else:  # Fail for convergence under infinite horizon
                P_check = 2
                break

        if np.min(np.linalg.eigvals(P_t)) < 0:
            # Cost matrix has negative eigenvalues - this should not happen
            raise Exception('Cost is not positive semi-definite')

        P = dc(P_t)












if __name__ == '__main__':
    print('functionfile_model.py code check complete')
