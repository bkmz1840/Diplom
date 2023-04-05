from scipy.optimize import minimize
from opt import *

import matplotlib.pyplot as plt


with open('input.txt') as f:
    data = f.read().split('\n')

T = 290
n = len(data)
T_r = 273
mu_r = 1
H_r = 100
ro_r = 10 ** 22
mu_0 = 1.256637
k_b = 1.380649
I = 6
mu_min = 0.000001
mu_step = 0.00001
step = 0.000001

input_data = []
for line in data:
    mapped_line = map(float, line.split(' '))
    point = list(mapped_line)
    input_data.append(point)

H = list(map(lambda e: e[0], input_data))
M = list(map(lambda e: e[1], input_data))

M_r = max(M)
T_w = T / T_r

H_w = [h / H_r for h in H]
M_w = [m / M_r for m in M]

a = 1000 / M_r # mu_r * ro_r = 10^(-19) * 10^22 = 1000
b = (mu_0 * 1 * 1) / (k_b * T_w * T_r) # Степени 10 сократились

mu_border = -10000
bounds = [
    [1, 10],
]
bounds.extend([[0, 1] for _ in range(I)])

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

    initial_guess = get_start_guess(I) #[1]
    print(f'Init guess: {initial_guess}')
    #initial_guess.extend(get_start_guess(I))

    print('Start minimization')
    res = minimize(
        lambda p: optimization_func(p, mus, n, a, b, H_w, M_w),
        initial_guess,
        bounds=bounds,
        constraints=constraints
    )
    
    print(f'Res success: {res.success}')
    if not res.success:
        not_success = True
        continue

    print(f'Res result: {res.fun}')
    found_params = list(res.x)[1:]
    result.append(OptimizationResult(found_params, mus, found_params[0]))

div1 = (7 * I + 3) // 2
div2 = div1 + 1
div3 = 8 * I - 1
header = "|" + " " * div1 + "mu" + " " * div2 + "| |" + " " * div2 + "p" + " " * div2 + "|"
splitter = "|" + "-" * div3 + "| |" + "-" * div3 + "|"
print(header)
print(splitter)
for r in result:
    print(r.form_line())
print(splitter)

H_t = []
M_t = []
for r in result:
    back_v = r.back(a, b, H_w)
    h_t = []
    m_t = []
    for backed_v in back_v:
        h, m = backed_v
        h_t.append(h)
        m_t.append(m)
    H_t.append(h_t)
    M_t.append(m_t)

plt.grid()
plt.plot(H_w, M_w, color="black")
for i in range(len(H_t)):
    plt.plot(H_t[i], M_t[i])
plt.show()

x = []
y = []
for r in result:
    x.extend(map(lambda e: round(e, 2), r.mus))
    y.extend(map(lambda e: round(e, 2), r.p))

plt.grid()
plt.scatter(x, y)
plt.show()
