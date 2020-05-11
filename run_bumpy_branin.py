# MIT License

# Copyright (c) 2020 Raphael Haftka, Charles Jekel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
import numpy as np
from scipy.optimize import differential_evolution as de

# Optimization parameters
seed_num = 1212
np.random.seed(seed_num)  # set random seed for Reproducibility
# setting the random seed makes all optimization runs repeatable
n_var = 2  # number of design variables
n_runs = 1000  # number of optimization runs
pop_size = 10  # DE population size
max_iter = 20  # number of DE iterations
use_bfgs = False  # whether to use BFGS after DE

# The number of function evaluations for DE is given as
# (max_iter + 1) * pop_size * n_var
# (20 + 1) * 10 * 2 = 420 function evaluations for the given parameters above

# bump information
bump_amp = 10.
opt_fun = 0.397887-bump_amp*np.exp(-1.)
tol_fun = 0.01  # how close we need to get to optimum to declare success
epsilon = 1
bumpid = 1
# coordinates of the three optima
x1 = [-np.pi, 12.275]
x2 = [np.pi, 2.275]
x3 = [9.42478, 2.475]
x0 = x1  # the coordinates of the bumped optimum
if bumpid == 2:
    x0 = x2
if bumpid == 3:
    x0 = x3

# Branin-Hoo paramters
a = 1.
b = 5.1 / (4*np.pi**2)
c = 5/np.pi
r = 6.
s = 10.
t = 1. / (8.*np.pi)


def bump_2d(x, x0, epsilon):
    r2 = (x[0]-x0[0])**2+(x[1]-x0[1])**2
    bump = np.exp(-1/(1-epsilon**2*r2)) * np.heaviside((1./epsilon**2-r2), 0.5)
    return bump


def my_fun(x):
    # define the bumpy Branin-Hoo function
    A = a*(x[1] - b*x[0]**2+c*x[0]-r)**2
    B = s*(1-t)*np.cos(x[0])+s
    # subtract a bump at the first global optimum
    bump1 = bump_2d(x, x0, epsilon)
    return A + B-bump_amp*bump1


# set up optimization bounds
bounds = np.zeros((n_var, 2))
bounds[0, 0] = -10.0  # lower bound for first variable
bounds[0, 1] = 10.0  # upper bound for first variable
bounds[1, 0] = -15.0  # lower bound for second variable
bounds[1, 1] = 15.0   # upper bound for second variable

# set up zeros for optimization results
res_design_vars = np.zeros((n_runs, n_var))
res_fun_values = np.zeros(n_runs)
successes = np.zeros(n_runs)
optid = np.zeros(n_runs)
dist = np.zeros(3)
# The fraction of runs in each optimum region
b_1 = 0.
b_2 = 0.
b_3 = 0.
# This will be the average number of function evaluations in N_runs
avg_nfev = 0
for i in range(n_runs):
    res = de(my_fun, bounds, maxiter=max_iter, popsize=pop_size,
             tol=0.0,  # this needs to be 0.0 to turn off convergence
             disp=False,  # this prints best objective value each iteration
             polish=use_bfgs,  # whether to run bfgs after
             init='latinhypercube'  # LHS random sampling for initial pop
             )

    res_x = res.x  # the optimum design point
    res_f = res.fun  # the optimum function value

    # find which minimum it went to, success, and store results
    dist[0] = np.linalg.norm(res_x-x1)*epsilon
    dist[1] = np.linalg.norm(res_x-x2)*epsilon
    dist[2] = np.linalg.norm(res_x-x3)*epsilon
    if dist[0] < 1:
        b_1 = b_1+1
    if dist[1] < 1:
        b_2 = b_2+1
    if dist[2] < 1:
        b_3 = b_3+1
    res_design_vars[i] = res_x
    res_fun_values[i] = res_f
    if opt_fun+tol_fun >= res_f:
        successes[i] = 1
    avg_nfev = avg_nfev + res.nfev
# cumulative number of function evals in in nruns
n_fvals = (max_iter + 1) * pop_size * n_var
cum_number_eval = res.nfev*n_runs
# number of successful runs based on tol_fun
num_success = np.sum(opt_fun+tol_fun >= res_fun_values)
# number of function evaluations
avg_nfev = avg_nfev/n_runs
b_1 = b_1/n_runs
b_2 = b_2/n_runs
b_3 = b_3/n_runs

pf = 100*(1-np.sum(successes)/n_runs)  # percent failures

print('Use bfgs?', use_bfgs)
print('Population', pop_size)
print('Max iterations', max_iter)
print('Percent failures', pf)
print('Average num function evals', avg_nfev)
print('Fraction at x1', b_1)
print('Fraction at x2', b_2)
print('Fraction at x3', b_3)
