from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

from opt_function import (
    constraint,
    get_initial_guess,
    optimization_func,
    M_func,
)

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def form_input(data):
    H = []
    M = []
    for line in data:
        H.append(float(line[0]))
        M.append(float(line[1]))
    
    return H, M


def get_mus(start, step, count):
    result = [start]
    
    current = start
    for _ in range(count - 1):
        current += step
        result.append(current)
    
    return result


def generate_mus(start, step, row_step, count):
    res = []
    border = None
    cur_start = start
    
    while True:
        mus = get_mus(cur_start, step, count)
        
        if border is None:
            border = mus[1]
        
        if mus[0] >= border:
            break
        
        res.append(mus)
        cur_start += row_step
    
    return res


def get_start_data(H_r, mu_0, k_b, T, evaluate_M=False):
    data = np.loadtxt('input.txt')
    
    H = data[:, 0]
    M = data[:, 1]
    
    M_r = np.max(M)
    
    H_w = [h / H_r for h in H]
    
    M_w = None
    if evaluate_M:
        M_w = [m / M_r for m in M]
    
    a = 10 ** 3 / M_r
    b = (mu_0 * 10 ** (-2) * H_r) / (k_b * T)
    
    return H_w, M_w, M_r, a, b


def draw_result_plot(H_w, M_w, backed_M, max_offsets=None, muses=None):
    n = len(backed_M)
    count_rows = n // 2 + n % 2
    fig, axes = plt.subplots(nrows=count_rows, ncols=2)
    
    cur_row = -1

    for i, back_M in enumerate(backed_M):
        if i % 2 == 0:
            cur_row += 1
        
        axes[cur_row, i % 2].grid()
        axes[cur_row, i % 2].plot(H_w, back_M)
        axes[cur_row, i % 2].scatter(H_w, M_w, color='black')
        
        max_offset = ''
        if max_offsets is not None:
            max_offset = f'. Max offset: {max_offsets[i]}'
        
        title = f'Try {i + 1}'
        if muses is not None:
            title = f'Mu_i: ' + ', '.join(map(lambda m: str(round(m, 2)), muses[i]))
        
        axes[cur_row, i % 2].set_title(title + max_offset)

    plt.tight_layout()
    plt.show()


def draw_plot(results, test_data=None):
    results.sort(key=lambda r: r[0])
    
    x = list(map(lambda r: r[0], results))
    y = list(map(lambda r: r[1], results))
    
    plt.grid()
    
    if test_data is not None:
        plt.scatter(test_data[0], test_data[1], color='black', marker="D")

    plt.scatter(x, y, color='red')
    plt.show()


def solve(args):
    H_r = args['H_r']
    mu_0 = args['mu_0']
    k_b = args['k_b']
    T = args['T']
    I = args['I']
    mu_min = args['mu_min']
    mu_step = args['mu_step']
    step = args['step']
    
    H_w, M_w, M_r, a, b = get_start_data(H_r, mu_0, k_b, T, evaluate_M=True)
    
    args['n'] = len(H_w)
    args['M_r'] = M_r
    
    muses = generate_mus(mu_min, mu_step, step, I)
    one_res_len = (1 + I + 1)
    res_arr_len = one_res_len * len(muses)  # (val + p_k + ro) * mu_k_j
    
    shared_arr = mp.RawArray(ctypes.c_double, res_arr_len)
    queue = mp.JoinableQueue()
    
    for i in range(len(muses)):
        queue.put(i)
    
    res_index = 0
    
    for i in range(len(muses)):
        worker = mp.Process(
            target=find_result_by_mus,
            args=(
                args,
                H_w, M_w, muses[i], a, b,
                shared_arr, res_index, queue
            )
        )
        worker.start()
        
        res_index += one_res_len
    
    queue.join()
    
    min_res = np.reshape(np.frombuffer(shared_arr), (len(muses), one_res_len))
    results = []
    backed_M = []
    
    for i, r in enumerate(min_res):
        mus = muses[i]
        val = r[0]
        x = r[1:]
        
        print(f'mu_i: {mus}')
        print(f'Min val: {val}')
        print(f'Min x: ro - {x[0]}; p_i - {x[1:]}')
        
        p = x[1:]
        for i in range(len(mus)):
            results.append((mus[i], p[i]))
        
        back_M = [M_func(x, a, b, mus, h) for h in H_w]
        backed_M.append(back_M)
    
    draw_result_plot(H_w, M_w, backed_M, muses=muses)
    draw_plot(results)


def find_result_by_mus(
    args,
    H_w, M_w, mus, a, b,
    res_arr, index, queue,
    init_guess_pos=None
):
    I = args['I']
    n = args['n']
    M_r = args['M_r']
    mu_0 = args['mu_0']
    H_r = args['H_r']
    k_b = args['k_b']
    T = args['T']

    interations_count = 50
    max_offset = 1

    try_number = 1
    
    print(f'Start try: {mus[0]}')

    while max_offset > 0.09:
        val, x, found_picked_places = get_minimization_result({
            'I': I,
            'n': len(M_w),
            'M_r': M_r,
            'mu_0': mu_0,
            'H_r': H_r,
            'k_b': k_b,
            'T': T,
        }, mus, H_w, M_w, interations_count, init_guess_pos=init_guess_pos)
        
        back_M = [M_func(x, a, b, mus, h) for h in H_w]
        
        max_offset = -1000
        for i in range(len(M_w)):
            offset = abs(M_w[i] - back_M[i])
            
            if offset > max_offset:
                max_offset = offset
        
        print(
            '*' * 11,
            f'Try {try_number}, ({mus[0]})',
            '-' * 11,
            f'Max offset: {max_offset}, ({mus[0]})',
            '*' * 11,
            sep='\n'
        )

        try_number += 1
        
        if try_number > 100 and max_offset < 0.2:
            break
    
    res_arr[index] = val
    
    for i in range(index + 1, index + 1 + len(x)):
        res_arr[i] = x[i - index - 1]
    
    print(f'\n\n\n TASK DONE ({mus[0]}) \n\n')
    queue.task_done()


def get_minimization_result(args, mus, H, M, interations_count, init_guess_pos=None):
    I = args['I']
    n = args['n']
    M_r = args['M_r']
    mu_0 = args['mu_0']
    H_r = args['H_r']
    k_b = args['k_b']
    T = args['T']

    a = 10 ** 3 / M_r
    b = (mu_0 * 10 ** (-2) * H_r) / (k_b * T)
    
    constraints = ({'type': 'eq', "fun": constraint})
    bounds = [
        [0, 100],
    ]
    bounds.extend([[0, 1] for _ in range(I)])
    opt_args = {
        'a': a,
        'b': b,
        'H': H,
        'M': M,
        'n': n,
        'mus': mus,
    }
    
    min_val = 10 ** 5
    min_x = None
    found_picked_places = None
    
    for _ in range(interations_count):
        while True:
            initial_guess, picked_places = get_initial_guess(
                bounds[0], I,
                init_guess_pos=init_guess_pos
            )

            res = minimize(
                optimization_func,
                initial_guess,
                bounds=bounds,
                constraints=constraints,
                args=(opt_args),
            )

            if res.success:
                x = list(map(float, res.x))
            
                if min_val > res.fun:
                    min_val = res.fun
                    min_x = x
                    found_picked_places = picked_places
                
                break
    
    return min_val, min_x, found_picked_places
