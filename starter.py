from main import optimize_main, get_minimization_result
from opt import form_input, M_func, get_mus
import numpy as np


T = 290
mu_r = 1
H_r = 100
ro_r = 10 ** 22
mu_0 = 1.256637
k_b = 1.380649
I = 2
mu_min = 0.1
mu_step = 0.1
step = 0.05


def main():
    data = np.loadtxt('input.txt')
    
    optimize_main(data, {
        "T": T,
        "mu_r": mu_r,
        "H_r": H_r,
        "ro_r": ro_r,
        "mu_0": mu_0,
        "k_b": k_b,
        "I": I,
        "mu_min": mu_min,
        "mu_step": mu_step,
        "step": step,
    })


def test():
    data = np.loadtxt('input.txt')
    
    H = data[:, 0]
    M = data[:, 1]
    
    print(H, M, sep='\n')
    
    M_r = np.max(M)
    
    H_w = [h / H_r for h in H]
    a = 10 ** 3 / M_r
    b = (mu_0 * 10 ** (-2) * H_r) / (k_b * T)
    
    test_params = [10, 0.8, 0.2]
    test_mus = [1, 10]
    
    M_w = [
        M_func(test_params, test_mus, a, b, h)
        for h in H_w
    ]
    print(f'M_w: {M_w}')
    
    # data = [f"{H_w[i] * H_r} {M_w[i] * M_r}" for i in range(len(H_w))]
    
    # print('Test data:')
    # print("\n".join(data))
    
    # optimize_main(data, {
    #     "T": T,
    #     "mu_r": mu_r,
    #     "H_r": H_r,
    #     "ro_r": ro_r,
    #     "mu_0": mu_0,
    #     "k_b": k_b,
    #     "I": I,
    #     "mu_min": mu_min,
    #     "mu_step": mu_step,
    #     "step": step,
    # }, M_r=M_r)
    result = get_minimization_result({
        "T": T,
        "mu_r": mu_r,
        "H_r": H_r,
        "ro_r": ro_r,
        "mu_0": mu_0,
        "k_b": k_b,
        "I": I,
        "M_r": M_r,
        "n": len(H_w),
    }, test_mus, H_w, M_w)
    print(f'Result: {result}')


if __name__ == "__main__":
    test()
