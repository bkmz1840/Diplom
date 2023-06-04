from functions import (
    solve,
    get_minimization_result,
    get_mus,
    get_start_data,
    draw_result_plot,
    draw_plot,
)
from opt_function import M_func

import numpy as np


# Args
T = 290
mu_0 = 1.256637
k_b = 1.380649
mu_r = 1
ro_r = 10 ** 22
H_r = 100
M_r = None  # максимум по M
n = None  # количество данных
I = 6
mu_min = 0.1
mu_step = 1
step = 0.2


def test():
    H_w, _, M_r, a, b = get_start_data(H_r, mu_0, k_b, T)
    
    test_mus = [1, 2, 3, 4, 5, 6]  # [mu_1, ..., mu_n]
    test_params = [5, 0.06, 0.15, 0.32, 0.35, 0.1, 0.02]  # [ro, p_1, ..., p_n]
    # test_mus = [1, 3, 5, 7, 9]
    # test_params = [5, 0.62, 0.21, 0.12, 0.03, 0.02]
    
    M_w = [M_func(test_params, a, b, test_mus, h) for h in H_w]
    interations_count = 25
    tries_count = 5

    backed_M = []
    max_offsets = []

    for _ in range(tries_count):
        val, x, __ = get_minimization_result({
            'I': I,
            'n': len(M_w),
            'M_r': M_r,
            'mu_0': mu_0,
            'H_r': H_r,
            'k_b': k_b,
            'T': T,
        }, test_mus, H_w, M_w, interations_count)

        print('*' * 11)
        print(f'Min val: {val}')
        print(f'Min x: ro - {x[0]}; p_i - {x[1:]}')

        max_offset = -1000
        for i in range(1, len(test_params)):
            offset = abs(test_params[i] - x[i])
            
            if offset > max_offset:
                max_offset = offset

        max_offsets.append(round(max_offset, 6))
        back_M = [M_func(x, a, b, test_mus, h) for h in H_w]
        backed_M.append(back_M)
    
    draw_result_plot(H_w, M_w, backed_M, max_offsets=max_offsets)


def test_with_offset():
    H_w, _, M_r, a, b = get_start_data(H_r, mu_0, k_b, T)
    
    # test_mus = [1, 2, 3, 4, 5, 6]  # [mu_1, ..., mu_n]
    # test_params = [5, 0.06, 0.15, 0.32, 0.35, 0.1, 0.02]  # [ro, p_1, ..., p_n]
    test_mus = [1, 2]
    test_params = [5, 0.8, 0.2]
    
    M_w = [M_func(test_params, a, b, test_mus, h) for h in H_w]
    
    interations_count = 25
    backed_M = []
    border_mu = test_mus[1]
    mus = test_mus
    cur_mu_start = test_mus[0]
    init_guess_pos = None
    results = []
    
    muses = []

    while True:
        print(f'Positions: {init_guess_pos}')
        
        val, x, found_picked_places = get_minimization_result({
            'I': I,
            'n': len(M_w),
            'M_r': M_r,
            'mu_0': mu_0,
            'H_r': H_r,
            'k_b': k_b,
            'T': T,
        }, mus, H_w, M_w, interations_count, init_guess_pos=init_guess_pos)

        print('*' * 11)
        print(f'mu_i: {mus}')
        print(f'Min val: {val}')
        print(f'Min x: ro - {x[0]}; p_i - {x[1:]}')
        print(f'Found picked places: {found_picked_places}')
        
        if init_guess_pos is None:
            init_guess_pos = found_picked_places
        
        muses.append(mus)
        
        p = x[1:]
        for i in range(len(mus)):
            results.append((mus[i], p[i]))

        back_M = [M_func(x, a, b, mus, h) for h in H_w]
        backed_M.append(back_M)

        cur_mu_start += step
        mus = get_mus(cur_mu_start, mu_step, I)
        
        if abs(border_mu - mus[0]) <= 10 ** (-6):
            break
    
    draw_result_plot(H_w, M_w, backed_M, muses=muses)
    draw_plot(results, test_data=[test_mus, test_params[1:]])


def main():
    args = {
        'T': T,
        'mu_0': mu_0,
        'k_b': k_b,
        'mu_r': mu_r,
        'ro_r': ro_r,
        'H_r': H_r,
        'I': I,
        'mu_min': mu_min,
        'mu_step': mu_step,
        'step': step,
    }
    
    solve(args)


if __name__ == '__main__':
    # test()  # Простой тест 
    # test_with_offset()  # Тест со сдвигом mu_i
    main()  # Реальные данные
