from mpmath import coth, log

import random
import numpy as np


def constraint(params):
    return 1 - np.sum(params[1:])


def get_initial_guess(bound_first, count, init_guess_pos=None):
    random.seed()
    
    guess = [None for _ in range(count + 1)]
    sum = 0
    
    a, b = bound_first
    guess[0] = round(random.uniform(a, b), 2)
    
    places = [i for i in range(1, count + 1)]
    if init_guess_pos is not None:
        places = init_guess_pos.copy()

    picked_places = []
    
    for _ in range(count - 1):
        p = round(random.uniform(0.0, 1.0 - sum), 2)
        sum += p
        
        if init_guess_pos is None:
            place = random.sample(places, 1)[0]
            places.remove(place)
        else:
            place = places.pop(0)

        picked_places.append(place)
        
        guess[place] = p
    
    last_place = places[0]
    guess[last_place] = round(1.0 - sum, 2)
    picked_places.append(last_place)
    
    return guess, picked_places


def L_func(z):
    if z == 0:
        return 0
    
    return coth(z) - 1 / z


def M_func(params, a, b, mus, H_k):
    sum = 0
    ro = params[0]
    p = params[1:]
    
    for k in range(len(p)):
        sum += p[k] * mus[k] * L_func(b * mus[k] * H_k)
    
    return a * ro * (10 ** 22) * sum


def R_func(z):
    return log(1 + z)


def optimization_func(params, args):
    n = args['n']
    H = args['H']
    M = args['M']
    a = args['a']
    b = args['b']
    mus = args['mus']
    
    sum = 0
    
    for i in range(n):
        if M[i] == 0:
            continue

        term = ((M[i] - M_func(params, a, b, mus, H[i])) / M[i]) ** 2
        sum += R_func(term)
    
    return sum
