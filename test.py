from scipy.optimize import minimize


def f(p):
    return 2 + p[0] + 3 + p[1]


def constraint(p):
    res = 1 - sum(p)
    
    print(f'Constr: {res}')
    
    return 1 - sum(p)


bounds = [
    [0, 1],
    [0, 1]
]
guess = [0.5, 0.5]
constraints = ({'type': 'eq', "fun": constraint})

res = minimize(f, guess, bounds=bounds, constraints=constraints)

print(res)
