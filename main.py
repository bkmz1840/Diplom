from scipy.optimize import minimize, Bounds, root
from opt import *

import matplotlib.pyplot as plt
import numpy as np


def optimize_main(data, params, M_r=None):
    T = params["T"]
    n = len(data)
    mu_r = params["mu_r"]
    H_r = params["H_r"]
    ro_r = params["ro_r"]
    mu_0 = params["mu_0"]
    k_b = params["k_b"]
    I = params["I"]
    mu_min = params["mu_min"]
    mu_step = params["mu_step"]
    step = params["step"]

    input_data = form_input(data)

    H = input_data[0]
    M = input_data[1]

    M_r = np.max(M) if M_r is None else M_r
    print(f'M_r: {M_r}')

    H_w = [h / H_r for h in H]
    M_w = [m / M_r for m in M]

    a = 1000 / M_r # mu_r * ro_r = 10^(-19) * 10^22 = 1000
    b = (mu_0 * 1 * 1) / (k_b * T) # Степени 10 сократились

    mu_border = -10000
    bounds = [
        [0, 10],
        [0, 10],
    ]
    bounds.extend([[0, 1] for _ in range(I)])
    print(f"Bounds: {bounds}")

    constraints = ({'type': 'eq', "fun": constraint})
    cur_mu_min = mu_min
    result = []
    not_success = False
    while True:
        if not not_success:
            mus = get_mus(cur_mu_min, mu_step, I)

            if mu_border == -10000:
                mu_border = mus[1]

            if abs(mu_border - mus[0]) <= 10 ** (-6):
                print(f'BREAK: {mu_border} --- {mus[0]}')
                break

            print(f'm: {mus}')
            cur_mu_min += step
        else:
            not_success = False

        initial_guess = get_start_guess(I)
        print(f'Init guess: {initial_guess}')

        print('Start minimization')
        fun = lambda p: optimization_func(p, mus, n, a, b, H_w, M_w)
        res = minimize(
            fun,
            initial_guess,
            bounds=bounds,
            constraints=constraints
        )
        
        print(f'Res result: {res.success}')
        if not res.success:
            not_success = True
            continue

        print(f'Res result: {res.fun}')
        found_params = list(res.x)[2:]
        ro = res.x[0] * 10 ** res.x[1]
        result.append(OptimizationResult(found_params, mus, ro))

    div = 8 * I - 1
    div1 = div // 2
    header = "|" + " " * div1 + "mu" + " " * (div1 - 1) + "| |" + " " * div1 + "p" + " " * div1 + "| |   ro  |"
    splitter = "|" + "-" * div + "| |" + "-" * div + "| |-------|"
    print(header)
    print(splitter)
    for r in result:
        print(r.form_line())
    print(splitter)

    H_t = []
    M_t = []
    backed_results = results_to_M(result, a, b, H_w, I)
    backed_results_data = []
    
    for h, m in backed_results:
        H_t.append(h)
        M_t.append(m)
        backed_results_data.append(f"{h} {m}")
    
    print("Backed results:")
    print("\n".join(backed_results_data))

    plt.grid()
    plt.plot(H_w, M_w, color="black")
    plt.plot(H_t, M_t, color="red")
    plt.show()

    x = []
    y = []
    for r in result:
        x.extend(map(lambda e: round(e, 2), r.mus))
        y.extend(map(lambda e: round(e, 2), r.p))

    plt.grid()
    plt.scatter(x, y)
    plt.show()
