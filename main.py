from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name
from scipy.optimize import minimize
from opt import *

import numpy as np

with open('input.txt') as f:
    data = f.read().split('\n')

T = 290
n = len(data)

print(f'T: {T}')
print(f'n: {n}')

input_data = []
for line in data:
    mapped_line = map(float, line.split(' '))
    point = list(mapped_line)
    input_data.append(point)

H = list(map(lambda e: e[0], input_data))
M = list(map(lambda e: e[1], input_data))

print(f'H: {H}')
print(f'M: {M}')

T_r = 273
mu_r = 10 ** (-19)
H_r = 100
ro_r = 10 ** 22
mu_0 = 1.256637 * 10 ** (-6)
k_b = 1.380649 * 10 ** (-23)

H_w = [h / H_r for h in H]
M_r = max(M) + 10 ** (-3)
M_w = [m / M_r for m in M]
T_w = T / T_r

print(f'H_w: {H_w}')
print(f'M_w: {M_w}')

print(f'T_w: {T_w}')

a = (ro_r * mu_r) / M_r
b = (mu_0 * mu_r * H_r) / (k_b * T_w)

I = 6
mu_min = -1.1
mu_max = 1.3
mu_step = 0.4

mu_border = -10000
optimization_iterations = []
optimizers = {}
initial_guess = [10, 0.16666, 0.16666, 0.16666, 0.16666, 0.16666]
bounds = [
    [-10 ** 10, 10 ** 10],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
]
cur_mu_min = mu_min
while True:
    mus = get_mus(cur_mu_min, mu_step, I)
    print(f'm: {mus}')
    
    if mu_border == -10000:
        mu_border = mus[1]
    
    if mu_border - 0.1 <= mus[0]:
        print(f'BREAK: {mu_border} --- {mus[0]}')
        break
    
    cur_mu_min += 0.1
    
    # optimizer_name = recommend(
    #     lambda p: optimization_func(p, mus, n, a, b, H_w, M_w),
    #     n_dim=6,
    #     n_trials=1500
    # )[-1][-1]
    # print(f"optimizer: {optimizer_name}")
    
    # if optimizer_name not in optimizers:
    #     optimizers[optimizer_name] = optimizer_from_name(optimizer_name)
    
    # optimizer_func = optimizers[optimizer_name]
    
    # best_val, best_x = optimizer_func(
    #     lambda p: optimization_func(p, mus, n, a, b, H_w, M_w),
    #     n_dim=6,
    #     n_trials=1500
    # )
    
    res = minimize(
        lambda p: optimization_func(p, mus, n, a, b, H_w, M_w),
        initial_guess,
        bounds=bounds
    )
    
    print(f"res: {res}")
