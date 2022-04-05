import numpy as np
import networkx as netx
from copy import deepcopy as dc
# import pickle
import warnings

# import matplotlib
import matplotlib.pyplot as plt
# # import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from matplotlib.ticker import MaxNLocator
#
# matplotlib.rcParams['axes.titlesize'] = 12
# matplotlib.rcParams['xtick.labelsize'] = 12
# matplotlib.rcParams['ytick.labelsize'] = 12
# matplotlib.rcParams['axes.labelsize'] = 12
# matplotlib.rcParams['legend.fontsize'] = 10
# matplotlib.rcParams['legend.title_fontsize'] = 10
# matplotlib.rcParams['legend.framealpha'] = 0.5
# matplotlib.rcParams['lines.markersize'] = 5
# # matplotlib.rcParams['image.cmap'] = 'Blues'
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['savefig.bbox'] = 'tight'
# matplotlib.rcParams['savefig.format'] = 'pdf'


def system_package1(A_in=None, B_in=None, S_in=None, X0_in=None, W_in=None, Q_in=None, R_in=None, system_label=None):

    A = dc(A_in)
    nx = np.shape(A)[0]

    # S: cardinality constraint on the actuator set
    if S_in is None:
        S = nx // 2
    else:
        S = dc(S_in)

    B = dc(B_in)
    if B is None:
        B = list_to_matrix(random_selection(np.arange(0, nx), S), nx)['matrix']
    nu = np.shape(B)[1]

    X0 = dc(X0_in)
    if X0 is None:
        metric_fn = metric0
        # no initial state => eigmax(P0)
    elif np.ndim(X0) == 1:
        metric_fn = metric1
        # metric = 1 => initial state vector => x0 * P0 * x0
    elif np.ndim(X0) == 2:
        metric_fn = metric2
        # metric = 2 => covariance of zero-mean initial state distribution => trace(P0 * X0)
    else:
        raise Exception('Check initial conditions X0 - affects cost metrics function')

    if W_in is None:
        W = np.zeros_like(A)
    else:
        W = dc(W_in)

    # covariance of zero-mean additive disturbance - set to matrix of 0s for no additive disturbance

    Q = dc(Q_in)
    if Q is None:
        Q = np.identity(nx)
    R = dc(R_in)
    if R is None:
        R = np.identity(nu)

    system_name = dc(system_label)

    system = {'A': A, 'B': B, 'S': S, 'X0': X0, 'W': W, 'Q': Q, 'R': R, 'name': system_name, 'cost_metric': metric_fn, 'sys_type': 1}
    return system


####################################################


def create_graph(nx_in, graph_type='cycle', p=None, self_loop=True):
    if graph_type not in ['cycle', 'path', 'ER', 'BA']:
        raise Exception('Check network type')

    nx = dc(nx_in)
    net_check = True

    G = None
    while net_check:
        if graph_type == 'cycle':
            G = netx.generators.classic.cycle_graph(nx)
        elif graph_type == 'path':
            G = netx.generators.classic.path_graph(nx)
        elif graph_type == 'ER':
            if p is None:
                print('Specify edge probability for ER-graph')
                return None
            else:
                G = netx.generators.random_graphs.erdos_renyi_graph(nx, p)
        elif graph_type == 'BA':
            if p is None:
                print('Specify initial network size for BA-graph')
                return None
            else:
                G = netx.generators.random_graphs.barabasi_albert_graph(nx, p)

        if netx.algorithms.components.is_connected(G):
            net_check = False

    if G is None:
        print('Error: Check graph generator')
        return None

    Adj = netx.to_numpy_array(G)
    if self_loop:
        Adj += np.identity(nx)

    e = np.max(np.abs(np.linalg.eigvals(Adj)))
    A = Adj / e

    return_values = {'A': A, 'eig_max': e, 'Adj': Adj}
    return return_values


#####################################################


def list_to_matrix(B_list, nx):
    B_matrix = np.zeros((nx, nx))
    B_list = B_list.astype(np.int64)
    for i in range(0, len(B_list)):
        B_matrix[B_list[i], i] = 1
    return_vals = {'matrix': B_matrix, 'list': B_list}
    return return_vals


####################################################


def matrix_to_list(B):
    B_list = []
    for i in range(0, np.shape(B)[0]):
        if np.max(B[:, i]):
            B_list.append(np.argmax(B[:, i]))
    S_list = np.array(B_list)

    return_values = {'matrix': B, 'list': S_list}
    return return_values


#####################################


def metric0(P_in, sys_in):
    # Metric for no initial state data => worst case cost over unit circle distribution of initial states
    return np.max(np.linalg.eigvals(P_in))


def metric1(P_in, sys_in):
    # Metric for initial state vector => cost for given initial state vector
    return sys_in['X0'].T @ P_in @ sys_in['X0']


def metric2(P_in, sys_in):
    # Metric for covariance of zero-mean initial state distribution => trace of cost scaled with covariance
    return np.trace(P_in @ sys_in['X0'])


#####################################


def solve_constraints_initializer(Sys_in=None):

    # Maximum cost metric bound - set None to look for convergence rather than
    J_max = 10**8

    # Minimum convergence accuracy for cost matrix over iterations
    P_accuracy = 10**(-4)

    # Time horizon of simulation/recursion
    T_max = 100

    if Sys_in is not None:
        Sys = dc(Sys_in)
        nx = np.shape(Sys['A'])[0]

        # T_sim = Length of simulation
        T_sim = 100

        # Disturbances

        W_sim = np.random.default_rng().multivariate_normal(np.zeros(nx), Sys['W'], T_sim)

        # Initial state
        if Sys['X0'] is None:
            x0 = 10 * np.random.rand(nx, 1)
        elif np.ndim(Sys['X0']) == 1:
            x0 = dc(Sys['X0'])
        elif np.ndim(Sys['X0']) == 2:
            x0 = np.random.default_rng().normal(np.zeros(nx), Sys['X0'])
        else:
            raise Exception('Check initial state vector conditions for simulation')
    else:
        T_sim = None
        W_sim = None
        x0 = None

    return_values = {'J_max': J_max, 'T_max': T_max, 'P_accuracy': P_accuracy, 'T_sim': T_sim, 'W_sim': W_sim, 'x0': x0}
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
    return None


#####################################


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            warnings.warn('Symmetry fail for positive definite matrix check', stacklevel=4)
            return False
    else:
        return False


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


def solve_cost_recursion(sys_in=None, solve_constraints=solve_constraints_initializer(), cost_fn=recursion_lqr_1step):

    # if solve_constraints_in is None:
    #     solve_constraints = solve_constraints_initializer()
    # else:
    #     solve_constraints = dc(solve_constraints_in)

    Sys = dc(sys_in)

    A = Sys['A']
    B = Sys['B']

    Q = Sys['Q']
    R = Sys['R']

    if np.shape(A)[1] != np.shape(Q)[0]:
        raise Exception('Check A vs Q dimensions - number of states do not match')
    if np.shape(B)[1] != np.shape(R)[0]:
        raise Exception('Check B vs R dimensions - number of inputs do not match')

    P = dc(Q)
    P_check = 0
    K_0 = None
    K_t = None

    t_steps = 0
    while not P_check:  # P_check = 0 for recursion to continue, 1 for successful exit, 2 for failure exit

        rec_calc = cost_fn(P, Sys)
        P_t = rec_calc['P']
        K_t = rec_calc['K']

        if t_steps == 0:
            K_0 = dc(K_t)

        if solve_constraints['J_max'] is not None and Sys['cost_metric'](P_t, Sys) > solve_constraints['J_max']:
            # Exceeding design cost bounds
            P_check = 2
            break
        elif solve_constraints['J_max'] is None and matrix_convergence_check(P_t, P, solve_constraints['P_accuracy'], None):
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

        if is_pos_def(P_t) < 0:
            # Sanity check for cost matrix with negative eigenvalues - this should not happen
            raise Exception('Cost is not positive definite')

        P = dc(P_t)
        t_steps += 1

    return_values = {'J': None, 't_steps': t_steps, 'K_0': K_0, 'K_t': K_t, 'P_check': P_check}
    if P_check == 1:  # Successful control with costs within J_max over T_max or converged costs in t_steps < T_max
        return_values['J'] = Sys['cost_metric'](P, Sys)
    elif P_check == 2:  # Failed control
        if solve_constraints['J_max'] is not None:  # finite horizon: cost metric at t_steps < T_max exceeds J_max
            return_values['J'] = solve_constraints['J_max']
        else:  # infinite horizon: cost metric has not converged within the time horizon
            return_values['J'] = np.inf

    return return_values


#####################################


def random_selection(S, n, p=None):
    #  Choose n from set S randomly with
    S_choice = np.random.default_rng().choice(S, n, False, p)
    return S_choice


#####################################


def greedy_actuator_selection(Sys_in, S=None, solve_constraints=solve_constraints_initializer(), cost_fn=recursion_lqr_1step):
    #  Select actuators for Sys under a cardinality constraint K
    Sys = dc(Sys_in)

    Sys['B'] = np.zeros_like(Sys['A'])
    nx = np.shape(Sys['A'])[0]

    # if solve_constraints_in is None:
    #     solve_constraints = solve_constraints_initializer()
    # else:
    #     solve_constraints = dc(solve_constraints_in)

    if 'S' not in Sys or Sys['S'] is None:
        Sys['S'] = nx

    B_list = np.arange(0, nx)
    J_rec = np.zeros(S)
    t_rec = np.zeros(S)
    for i in range(0, S):
        # print('Step:', i)
        # print('B_list:', B_list)
        J_test = np.zeros(len(B_list))
        t_test = np.zeros(len(B_list))
        for j in range(0, len(B_list)):
            Sys_test = dc(Sys)
            Sys_test['B'] = list_to_matrix(np.append(matrix_to_list(Sys_test['B'])['list'], B_list[j]), nx)['matrix']
            solve_test = solve_cost_recursion(Sys_test, solve_constraints, cost_fn)
            J_test[j] = dc(solve_test['J'])
            t_test[j] = dc(solve_test['t_steps'])

        greedy_s = np.argmin(J_test)
        if (solve_constraints['J_max'] is not None and J_test[greedy_s] == solve_constraints['J_max']) or np.isinf(J_test[greedy_s]):
            greedy_s = np.argmax(t_test)
        J_rec[i] = J_test[greedy_s]
        t_rec[i] = t_test[greedy_s]

        # print('s:', B_list[greedy_s])
        # print('J:', J_test)
        # print('t:', t_test)
        Sys['B'] = list_to_matrix(np.append(matrix_to_list(Sys['B'])['list'], B_list[greedy_s]), nx)['matrix']
        B_list = np.delete(B_list, greedy_s)

    return_values = {'J': J_rec, 't': t_rec, 'System': Sys}
    return return_values


#######################################################


def lqr_dynamics_cost_update(Sys, x0, K, sim_noise=None):

    u0 = K@x0
    if sim_noise is None or sim_noise['w'] is None:
        w = np.zeros(np.shape(Sys['A'])[0])
    else:
        w = dc(sim_noise['w'])
    # print(np.shape(Sys['A']@x0))
    # print(np.shape(Sys['B']@u0))
    # print(np.shape(w))
    x1 = Sys['A']@x0 + Sys['B']@u0 + w

    J0 = x0.T@Sys['Q']@x0 + u0@Sys['R']@u0

    return_values = {'u0': u0, 'x1': x1, 'J0': J0}
    return return_values

#######################################################


def simulate_system(Sys_in, K_type=1, solve_constraints=None):

    return_values = {}

    Sys = dc(Sys_in)
    nx = np.shape(Sys['A'])[0]
    return_values['nx'] = dc(nx)

    if solve_constraints is None:
        solve_constraints = solve_constraints_initializer(Sys)

    x_trajectory = np.zeros((nx, solve_constraints['T_sim']+1))
    x_trajectory[:, 0] = dc(solve_constraints['x0'])

    u_trajectory = np.zeros((nx, solve_constraints['T_sim']))
    J_trajectory = np.zeros(solve_constraints['T_sim']+1)
    warning_trajectory = np.ones(solve_constraints['T_sim']+1)
    # records 1 for no warnings or 0 if gain/actuator set is not feasible/bounded

    if K_type not in [1, 2, 3, 4]:
        raise Exception('Pick suitable gain type/test model')
    elif K_type == 1:
        return_values['label'] = 'Design-time random actuators and gain'
    elif K_type == 2:
        return_values['label'] = 'Design-time greedy actuators and gain'
    elif K_type == 3:
        return_values['label'] = 'Run-time greedy actuators and gain'
    elif K_type == 4:
        return_values['label'] = 'Run-time greedy actuators and gain with state information'

    if K_type == 1:  # Design-time random actuators and gain
        Sys['B'] = list_to_matrix(random_selection(np.arange(0, nx), 5), nx)['matrix']
        cost_solver = solve_cost_recursion(Sys, solve_constraints, recursion_lqr_1step)
        K = cost_solver['K_t']
        if cost_solver['P_check'] != 1:
            warnings.warn('Gain does guarantee finite costs or feasible control')
            warning_trajectory = np.zeros(solve_constraints['T_sim']+1)
    if K_type == 2:  # Design-time greedy best actuator, run-time gain
        Sys = dc(greedy_actuator_selection(Sys, Sys['S'], solve_constraints, recursion_lqr_1step)['System'])

    # Simulation loop
    for t in range(0, solve_constraints['T_sim']):

        if K_type in [3, 4]:  # Run-time actuator and gains
            if K_type == 4:  # Run-time state information update
                Sys['X0'] = dc(x_trajectory[:, t])
                Sys['cost_metric'] = metric1
            Sys = greedy_actuator_selection(Sys, Sys_in['S'], solve_constraints, recursion_lqr_1step)['System']
            warning_trajectory[t] = 0

        if K_type in [2, 3, 4]:  # run-time gain evaluation
            cost_solver = solve_cost_recursion(Sys, solve_constraints, recursion_lqr_1step)
            K = cost_solver['K_t']
            if cost_solver['P_check'] != 1:
                warnings.warn('Gain does guarantee finite costs or feasible control')
                warning_trajectory[t] = 0

        sim_noise = {'w': dc(solve_constraints['W_sim'][t, :])}
        try:
            update_results = lqr_dynamics_cost_update(Sys, x_trajectory[:, t], K, sim_noise)
        except NameError:
            raise Exception('Check K')

        x_trajectory[:, t+1] = dc(update_results['x1'])
        u_trajectory[:, t] = dc(update_results['u0'])
        J_trajectory[t+1] = dc(update_results['J0'])+J_trajectory[t]

    return_values = return_values | {'x': x_trajectory, 'u': u_trajectory, 'J': J_trajectory, 'check': warning_trajectory, 'T_sim': solve_constraints['T_sim']}
    return return_values


#######################################################


def plot_trajectory(data):
    fig1 = plt.figure(tight_layout=True)
    gs1 = GridSpec(3, 1, figure=fig1)

    ax1 = fig1.add_subplot(gs1[0, 0])
    for i in range(0, data['nx']):
        ax1.stairs(data['x'][i, :])
    ax1.set_title('States')

    ax2 = fig1.add_subplot(gs1[1, 0])
    for i in range(0, data['nx']):
        ax2.stairs(data['u'][i, :])
    ax2.set_title('Inputs')

    ax3 = fig1.add_subplot(gs1[2, 0])
    ax3.plot(np.arange(0, data['T_sim']+1), data['J'])
    ax3.set_title('Cost')

    plt.suptitle(data['label'])
    plt.show()
    return None

#######################################################


if __name__ == '__main__':
    print('functionfile_model.py code check complete')
