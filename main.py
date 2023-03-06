from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name
import numpy as np

def get_mus(start, step):
    res = [start]

    cur_mu = start
    for i in range(I - 1):
        cur_mu += step
        res.append(cur_mu)

    return res


def f(params):
    x, y = params
    return x ** 2 + y ** 2


optimizer = recommend(f, n_dim=2, n_trials=1500)[-1][-1]
optimizer_func = optimizer_from_name(optimizer)
print(optimizer_func(f, n_dim=2, n_trials=1500))
