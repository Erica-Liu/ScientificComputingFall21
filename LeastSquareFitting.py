#  Problem 2 in Homework 5
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Least Square Fitting

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import scipy.linalg as linalg
import numpy.linalg as la

# generate rates for each chemical
# return narray of rates
def gen_rates(low, high, m):
    return np.random.uniform(low, high, m)

# generate initial concentration values for each chemical
# return narray of rates
def gen_init_conts(low, high, m):
    return np.random.uniform(low, high, m)

# generate the time
def gen_times(low, high, n):
    return np.linspace(low, high, n)

# generate residuals with mean and stdev
def gen_residuals(mu, sigma, n):
    return np.random.normal(mu, sigma, n)

# generate fake observations
# F = f(t_j) + \xi_j
def gen_fake_obs(n, m, times, rates, init_conts, residuals):
    fake_obs = np.empty(n, dtype=np.float64)
    for j in range(n):
        fake_obs[j] = 0
        for i in range(m):
            fake_obs[j] += init_conts[i] * np.power(np.e, -rates[i] * times[j])
        fake_obs[j] += residuals[j]
    return fake_obs

# linear least square problem

# create M, b
def gen_M_b(n, m, times, rates, fake_obs):
    M = np.zeros((n,m))
    for j in range(n):
        for i in range(m):
            M[j,i] = np.power(np.e, -rates[i] * times[j])
    return M, fake_obs

# Using Cholesky decomposition
def cholesky_least_square_fitter(M, b):
    c, low = cho_factor(M.T @ M)
    return cho_solve((c, low), M.T @ b)

# Using QR decomposition
def QR_least_square_fitter(M,b):
    Q, R = linalg.qr(M)
    b_prime = Q.T @ b
    m = M.shape[1] # number of columns
    return linalg.solve(R[:m,], b_prime[:m])

# Using SVD
def SVD_least_square_fitter(M, b):
    U, sigma, VT = la.svd(M)
    m = M.shape[1]
    Sigma_pinv = np.zeros(M.shape).T
    Sigma_pinv[:m, :m] = np.diag(1 / sigma[:m])
    return VT.T @ Sigma_pinv @ U.T @ b

if __name__ == '__main__':
    #
    m = 3
    n = 5
    mu = 0.
    sigma = 1.
    # generate data
    rates = gen_rates(0., 1., m)
    init_conts = gen_init_conts(0.,1., m)
    times = gen_times(0.1, 5., n)
    residuals = gen_residuals(mu, sigma, n)
    fake_obs = gen_fake_obs(n, m, times, rates, init_conts, residuals)
    M, b = gen_M_b(n, m, times, rates, fake_obs)
    print(times)
    print(fake_obs)
    print(M)
    print(b)

    # least square fitting
    x_Cholesky = cholesky_least_square_fitter(M, b)
    x_QR = QR_least_square_fitter(M, b)
    x_SVD = SVD_least_square_fitter(M, b)

    print(x_Cholesky)
    print(x_QR)
    print(x_SVD)

    print(la.norm((M @ x_SVD) - b, 2))


