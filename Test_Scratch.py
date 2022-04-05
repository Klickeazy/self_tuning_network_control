import numpy as np

from functionfile_model import system_package1, solve_cost_recursion, random_selection, greedy_actuator_selection, create_graph, list_to_matrix, matrix_to_list, simulate_system, plot_trajectory, solve_constraints_initializer, plot_trajectory_comparisons

nx = 50
p = 0.4

A = 1.05*create_graph(nx, 'ER', p)['A']
# B = list_to_matrix(random_selection(np.arange(0, nx), 5), nx)['matrix']
S = 2
X0 = 10*np.random.rand(nx)

W = np.identity(nx)

# Sys = system_package1(A, B, X0_in=X0, W_in=W)
Sys = system_package1(A, S_in=S, X0_in=X0, W_in=W)
# print(matrix_to_list(Sys['B']))
# print(Sys)
# print('Rand B:\n', Sys['B'])

# results = solve_cost_recursion(Sys)
# print(results)
#
# results_greedy_selection = greedy_actuator_selection(Sys, 5)
# print(results_greedy_selection['J'])
# print(results_greedy_selection['t'])
# # print('Greedy B:', results_greedy_selection['System']['B'])

solve_constraints = solve_constraints_initializer(Sys)
simulate_results = {}
for i in [1, 4]:
    simulate_results[i] = simulate_system(Sys, i, solve_constraints)
    # plot_trajectory(simulate_results[i])
plot_trajectory_comparisons(simulate_results)

print('Done')