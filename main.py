from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name
from scipy.optimize import minimize
from opt import *

import numpy as np

with open('input.txt') as f:
    data = f.read().split('\n')

T = 290
n = len(data)

input_data = []
for line in data:
    mapped_line = map(float, line.split(' '))
    point = list(mapped_line)
    input_data.append(point)

H = list(map(lambda e: e[0], input_data))
M = list(map(lambda e: e[1], input_data))

T_r = 273
mu_r = 1
H_r = 100
ro_r = 10 ** 22
mu_0 = 1.256637
k_b = 1.380649

H_w = [h / H_r for h in H]
M_r = max(M) + 10 ** (-3)
M_w = [m / M_r for m in M]
T_w = T / T_r

a = 1000 / M_r # mu_r * ro_r = 10^(-19) * 10^22 = 1000
b = (mu_0 * 1 * 1) / (k_b * T_w) # Степени 10 сократились

I = 6
mu_min = -1.1
mu_max = 1.3
mu_step = 0.4

mu_border = -10000
optimization_iterations = []
optimizers = {}
bounds = [
    [1, 1.1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
]
constraints = ({'type': 'eq', "fun": constraint})
cur_mu_min = mu_min
result = []
while True:
    mus = get_mus(cur_mu_min, mu_step, I)
    print(f'm: {mus}')
    
    if mu_border == -10000:
        mu_border = mus[1]
    
    if mu_border - 0.1 <= mus[0]:
        print(f'BREAK: {mu_border} --- {mus[0]}')
        break
    
    cur_mu_min += 0.1

    initial_guess = [1]
    initial_guess.extend(get_start_guess(6))

    res = minimize(
        lambda p: optimization_func(p, mus, n, a, b, H_w, M_w),
        initial_guess,
        bounds=bounds,
        constraints=constraints
    )

    found_params = list(res.x)[1:]
    p_I = 1 - sum(found_params)
    found_params.append(p_I)
    result.append(OptimizationResult(found_params, mus))

header = "|" + " " * 22 + "mu" + " " * 23 + "| |" + " " * 23 + "p" + " " * 23 + "|"
splitter = "|" + "-" * (7 * 6 + 5) + "| |" + "-" * (7 * 6 + 5) + "|"
print(header)
print(splitter)
for r in result:
    print(r.form_line())
print(splitter)
