from main import optimize_main
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
    
    H = data[:, 0][1:3]
    H_w = H/ H_r
    M_r = 4428.242
    
    a = 1000 / M_r
    b = (mu_0 * 1 * 1) / (k_b * T)
    
    print(f"H_w: {H_w}")
    print(f"a: {a}\nb: {b}")
    
    params1 = [100, 0.8, 0.2]
    params2 = [100, 0.9, 0.1]
    M_w = [
        M_func(params1, get_mus(mu_min, mu_step, I), a, b, H_w[0]),
        M_func(params2, get_mus(mu_min + step, mu_step, I), a, b, H_w[1]),
    ]
    
    # data = [f"{H_w[i] * H_r} {M_w[i] * M_r}" for i in range(len(H_w))]
    new_data = np.empty((H_w.shape[0], 2))
    new_data[:, 0] =  H_w * H_r
    new_data[:, 1] =  np.array(M_w) * M_r
    
    print('Test data:')
    print(new_data)
    
    optimize_main(new_data, {
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
    }, M_r=M_r)


if __name__ == "__main__":
    # main()
    test()
