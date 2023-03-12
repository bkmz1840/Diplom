from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name

from mpmath import coth, log
from scipy.optimize import shgo, dual_annealing

import numpy as np


class OptimizationResult:
    def __init__(self, p, mus):
        self.p = p
        self.mus = mus


def L(z):
    if z == 0:
        return 0
    return coth(z) - 1/z


def get_sum(arr):
    sum = 0
    for e in arr:
        sum += e
    return e


def M_func(params, mus, a_arg, b_arg, H_i):
    sum = 0
    ro = params[0]
    p = params[1:]

    p_I = 1 - get_sum(p)
    p = np.append(p, p_I)

    for k in range(len(p)):
        sum += p[k] * mus[k] * L(b_arg * mus[k] * H_i)
    
    return a_arg * ro * sum


def get_mus(start, step, I):
    res = [start]

    cur_mu = start
    for i in range(I - 1):
        cur_mu += step 
        res.append(cur_mu)

    return res


def R(z):
    return log(1 + z)


def optimization_func(params, mus, n, a, b, H, M):
    sum = 0
    
    for i in range(n):
        term = (M_func(params, mus, a, b, H[i]) - M[i]) ** 2
        sum += R(term)

    return sum
