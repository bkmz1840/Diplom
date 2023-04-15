from scipy.optimize import minimize
from GPyOpt.methods import BayesianOptimization

import random


def f(p):
    vars = p[0]
    return 2 * vars[0] + 3 * vars[1]


def constraint(p):
    res = 1 - sum(p)
    
    print(f'Constr: {res}')
    
    return 1 - sum(p)


def get_start_guess(count):
    random.seed()
    guess = []
    cur_sum = 0
    
    for i in range(count - 1):
        cur_param = round(random.uniform(0.0, 1.0 - cur_sum), 1)
        cur_sum += cur_param
        guess.append(cur_param)
    
    guess.append(round((1.0 - cur_sum), 1))
    return guess


bounds = [
    (0, 1),
    (0, 1)
]
guess = get_start_guess(2)
constraints = ({'type': 'eq', "fun": constraint})

domain = [
    {'name': 'x', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'y', 'type': 'continuous', 'domain': (0, 1)},
]

myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=15)
myBopt.plot_acquisition()

# res = gp_minimize(f, bounds, n_calls=50)

# print(res)