from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name

from mpmath import coth, log
from scipy.optimize import shgo, dual_annealing

import numpy as np
import math


class OptimizationResult:
    def __init__(self, p, mus):
        self.p = p
        self.mus = mus

    def form_cell(self, num):
        divider = (7 - len(str(num))) / 2
        if round(divider) == divider:
            d = int(divider)
            return f"{' ' * d}{num}{' ' * d}"

        l = int(math.ceil(divider))
        r = int(math.floor(divider))
        return f"{' ' * l}{num}{' ' * r}"
    
    def form_line(self):
        res = []
        for m in self.mus:
            res.append(f"|{self.form_cell(round(m, 2))}")
        res[-1] += "| "
        
        for p in self.p:
            res.append(f"|{self.form_cell(round(p, 2))}")
        res[-1] += "|"
        return ''.join(res)
    
    def __str__(self):
        p_str = map(str, self.p)
        m_str = map(str, self.mus)
        res = "*" * 5 + "\n" \
            + ', '.join(m_str) + '\n' \
            + ', '.join(p_str) + '\n' \
            + "*" * 5
        return res


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
