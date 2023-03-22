from scipy.optimize import minimize

import random


def f(p):
    print(f'p: {p}')
    return 2 * p[0] + 3 * p[1]


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
    [0, 1],
    [0, 1]
]
guess = get_start_guess(2)
constraints = ({'type': 'eq', "fun": constraint})

res = minimize(f, guess, bounds=bounds, constraints=constraints)

print(res)