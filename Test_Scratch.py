import numpy as np

from functionfile_model import system_package1, solve_cost_recursion, random_selection, greedy_actuator_selection, create_graph, list_to_matrix

nx = 10
p = 0.4

A = create_graph(nx, 'ER', p)['A']
B = list_to_matrix(random_selection(np.arange(0, nx), 5), nx)['matrix']

X0 = 5*np.random.rand(nx)

Sys = system_package1(A, B, X0)
# print('Rand B:\n', Sys['B'])

results = solve_cost_recursion(Sys)
print(results)

results_greedy_selection = greedy_actuator_selection(Sys, 5)
print(results_greedy_selection['J'])
print(results_greedy_selection['t'])
# print('Greedy B:', results_greedy_selection['System']['B'])
