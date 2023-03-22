from humpday import recommend
from humpday.optimizers.alloptimizers import optimizer_from_name

from mpmath import coth, log
from scipy.optimize import shgo, dual_annealing

import numpy as np
import math
import random


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


def get_sum(p):
    sum = 0
    for e in p:
        sum += e
    return sum


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


def M_func(params, mus, a_arg, b_arg, H_i):
    sum = 0
    m = params[0]
    p = params[1:]
    sum_m = 0

    for k in range(len(p)):
        sum += p[k] * mus[k] * L(b_arg * mus[k] * H_i)
        sum_m = p[k] * mus[k]

    return a_arg * (m / sum_m) * sum


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


def constraint(p):
    return 1 - get_sum(p[1:])
