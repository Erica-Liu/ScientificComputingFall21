#  Problem 2 in Homework 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Least Square Fitting

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import scipy.linalg as linalg
import numpy.linalg as la

# generate rates for each chemical
# return narray of rates
def gen_rates(low, high, m):
    #return np.linspace(low, high, m)
    return np.random.uniform(low, high, m)

# generate initial concentration values for each chemical
# return narray of rates
def gen_init_conts(low, high, m):
    return np.random.uniform(low, high, m)

# generate the time
def gen_times(low, high, n):
    #return np.linspace(low, high, n)
    return np.random.uniform(low, high, n)

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

    # print the condition number
    condNum = max(sigma)/min(sigma)

    return VT.T @ Sigma_pinv @ U.T @ b, condNum

if __name__ == '__main__':
    n = 1000
    mu = 0.
    sigma = 20.
    for m in range(3,18):
        # generate data
        rates = gen_rates(0., 20., m)
        init_conts = gen_init_conts(0.,10., m)
        times = gen_times(0.1, 10., n)
        residuals = gen_residuals(mu, sigma, n)
        fake_obs = gen_fake_obs(n, m, times, rates, init_conts, residuals)

        # create matrix M and vector b
        M, b = gen_M_b(n, m, times, rates, fake_obs)

        # problem (d) harder problems
        x_SVD, condNum = SVD_least_square_fitter(M, b)
        try:
            cholesky_fail_flag = False
            x_Cholesky = cholesky_least_square_fitter(M, b)  # the small dimension

        except:
            #print("cholesky doesn't work")
            cholesky_fail_flag = True

        x_QR = QR_least_square_fitter(M, b)


        if cholesky_fail_flag:
            cho_diff = float("inf")
        else:
            cho_diff = la.norm(init_conts - x_Cholesky, ord=2)
        svd_diff = la.norm(init_conts - x_SVD, ord=2)
        qr_diff = la.norm(init_conts - x_QR, ord=2)

        Info = "condition number : {cond:12.5e}| cholesky_diff : {cho_diff:12.5e} | SVD_diff : {svd_diff:12.5e} | QR_diff : {qr_diff:10.5e} "

        Info = Info.format(cho_diff=cho_diff,svd_diff=svd_diff,qr_diff=qr_diff, cond= condNum)
        print(Info)

        """
        # problem (b) accuracy
        diff = la.norm(init_conts - x_SVD, ord=2)
        print("---------------test {i}----------------".format(i=i))
        svd_Info = "SVD fitter estimate:      {x_es:65}, with condition number {cond:10.10f}"
        svd_Info = svd_Info.format(x_es=str(x_SVD), cond=condNum)
        print(svd_Info)
        cho_Info = "Cholesky fitter estimate: {x_es:65}"
        cho_Info = cho_Info.format(x_es=str(x_Cholesky))
        print(cho_Info)
        qr_Info =  "QR fitter estimate:       {x_es:65}"
        qr_Info = qr_Info.format(x_es=str(x_QR))
        print(qr_Info)
        """

    #(c)
    # least square fitting



