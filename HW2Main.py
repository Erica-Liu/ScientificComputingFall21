#  Python 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Local Analysis

import numpy as np
from Integrator import RR1f, MR2f, RR1a, SR4a
import Integratee
from tabulate import tabulate

"""
Different functions
"""
def func(x):
    return np.power(x, 2.0)

def func1(x):
    return np.power(x, -0.5)

def func2(x):
    return np.power(x, 0.5)

def func3(x):
    return np.power(x, 1.5)

def func_e(x):
    lbd = 2.0
    return lbd * np.exp(-lbd * x)

def func_e_f(x):
    if x < 1:
        return np.exp(x)
    else:
        return np.power(x, 3.0)

def func_f(x, t):
    t = 2.0**10
    return np.cos(t * x * x)

# driving program
def numeric_integrate(test_arg):
    a, b, testee, trueValue, integrator = test_arg[0], test_arg[1], test_arg[2], test_arg[3], test_arg[4]
    table = []
    for n in range(1, 100, 10):
        est = integrator(a, b, n, testee)
        est2 = integrator(a, b, 2 * n, testee)
        err = est - trueValue
        err2 = est2 - trueValue
        asym_error = err * n
        row = [n, est, est2, err, asym_error, err2, err / err2]
        table.append(row)
    print(tabulate(table, headers=["n", "A(n)", "A(2n)", "E(n)", "E(n)/(1/n)", "E(2n)", "E(n)/E(2n)"], floatfmt='3.5e'))

if __name__ == '__main__':
    # problem 4.(a)(b)(c)
    testee = Integratee.Integratee(func)    # f = x^2
    testRR = [0, 1, testee, 1/3, RR1f]      # Rectangular Rule 1st order fixed method
    testMR = [0, 1, testee, 1/3, MR2f]      # Mid-point Rule 2nd order fixed method

    numeric_integrate(testRR)
    numeric_integrate(testMR)

    # Problem 4(d)
    testee1 = Integratee.Integratee(func1)
    test1 = [0, 1, testee1, 2, MR2f]
    testee2 = Integratee.Integratee(func2)
    test2 = [0, 1, testee2, 2/3, MR2f]
    testee3 = Integratee.Integratee(func3)
    test3 = [0, 1, testee3, 2/5, MR2f]

    numeric_integrate(test1)
    numeric_integrate(test2)
    numeric_integrate(test3)

    # problem (e)
    testee_e = Integratee.Integratee(func_e)
    n_e = 3
    tol = np.power(2.0, -23)
    #result = RR1a(0, 1, n_e, testee_e, tol)
    print("\nproblem 4(e): adaptive code RR1a")
    #print(result)

    testee_e_f = Integratee.Integratee(func_e_f)
    tol = np.power(2.0, -23)
    #result = RR1a(0, 2, n_e, testee_e, tol)
    print("\nproblem 4(e): adaptive code RR1a not-workng example")
    #print(result)

    # problem (f)
    print("\nproblem 4(f): adaptive code RR1a")
    table = []
    for t in [2.0** k for k in range(10, 23)]:
        def func_fs(x):
            return  np.cos(t * x * x)
        testee_f = Integratee.Integratee(func_fs)
        n_f = 3
        tol = np.power(2.0, -23)
        result = RR1a(0, 1, n_f, testee_f, tol)
        row = [t, result[0], result[1], result[2]]
        table.append(row)
    print(tabulate(table, headers=["t", "A(n)", "n", "Requested error achieved?"], floatfmt='3.5e'))

    # problem (g)
    print("\nproblem 4(g): adaptive code SR4a")
    table = []
    for t in [2.0 ** k for k in range(10, 23)]:
        def func_gs(x):
            return np.cos(t * x * x)
        testee_g = Integratee.Integratee(func_gs)
        n_g = 3
        tol = np.power(2.0, -23)
        result = SR4a(0, 1, n_g, testee_g, tol)
        row = [t, result[0], result[1], result[2]]
        table.append(row)
    print(tabulate(table, headers=["t", "A(n)", "n", "Requested error achieved?"], floatfmt='3.5e'))



