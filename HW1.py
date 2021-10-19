#  Python 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Source of Error

import numpy as np
import math
from tabulate import tabulate

# Problem 2: two ways of evaluating (x-1)^n

# Evaluate (x-1)^n with direct formula
# Find the relative error to illustrate accuracy when n >= 10, |x-1| <= 1/2
print("\n--Problem 2--")
n = 10
xValues = [1 + 0.05 * i for i in range(-10, 10)]
print(xValues)
table = []
for x in xValues:
    f = pow(x - 1, n)

    g = 0
    for k in range(n + 1):
        g += math.comb(n, k) * pow(x, n - k) * pow(-1, k)

    try:
        diff = abs(g - f)
        rdiff = diff / g
    except ZeroDivisionError:
        print("Zero Division error")
    row = [x, f, g, diff, rdiff]
    table.append(row)
print(tabulate(table, headers=["x", "f", "g", "Diff", "Relative Diff"], floatfmt='3.13f'))

# Problem 3: Tradeoff between roundoff error and truncation error
print("\n--Problem 3--")  # question: if I use pow then float, do I loss accuracy?
dDeltaXValues = [np.float64(pow(2, x)) for x in range(-1, -30, -2)]  # take double precision deltas from 2^-1
sDeltaXValues = [np.float32(pow(2, x)) for x in range(-1, -30, -2)]  # take single precision deltas from 2^-1

print("\nFor f(x) = e^x, at x = 1")
table = []  # output table initiation
dx = np.float64(1)  # x = 1
sx = np.float32(1)  # x = 1

for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    # calculate Dplus with two points forward difference approximation
    dDplus = (np.exp(dx + dDeltaX) - np.exp(dx)) / dDeltaX
    sDplus = (np.exp(sx + sDeltaX) - np.exp(sx)) / sDeltaX

    # actual value for the derivative of f at x = 1
    dFprime = np.e
    sFprime = np.float32(np.e)

    #absolute error
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)

    #relative error
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime

    # result format
    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError, sRelativeError]
    table.append(row)

print(tabulate(table,
               headers=["D delta x", "f'", "D+(double)", "AbsoluteErr(D)", "RelativeErr(D)", "S Delta x", "D+(single)",
                        "AbsoluteErr(s)", "RelativeErr(s)"], floatfmt="3.14f"))

print("\nf(x) = sin(x), at x = 1")
dx = np.float64(0) # x = 0
sx = np.float32(0) # x = 0
table = []
for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    # calculate Dplus with two points forward difference approximation
    dDplus = (np.sin(dx + dDeltaX) - np.sin(dx)) / dDeltaX      # f = sinx
    sDplus = (np.sin(sx + sDeltaX) - np.sin(sx)) / sDeltaX

    # f' = 1
    dFprime = np.float64(1)
    sFprime = np.float32(1)

    #absolute error
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)

    #relative error
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime
    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError, sRelativeError]
    table.append(row)

print(tabulate(table,
               headers=["D delta x", "f'", "D+(double)", "AbsoluteErr(D)", "RelativeErr(D)", "S Delta x", "D+(single)",
                        "AbsoluteErr(s)", "RelativeErr(s)"], floatfmt="3.14f"))

print("\nf(x) = x^2, at x = 1")
dx = np.float64(1)
sx = np.float32(1)
table = []
for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    dDplus = (pow(dx + dDeltaX, 2) - pow(dx,2)) / dDeltaX
    sDplus = (pow(sx + sDeltaX, 2) - pow(sx, 2)) / sDeltaX
    dFprime = np.float64(2)
    sFprime = np.float32(2)
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime
    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError, sRelativeError]
    table.append(row)

print(tabulate(table,
               headers=["D delta x", "f'", "D+(double)", "AbsoluteErr(D)", "RelativeErr(D)", "S Delta x", "D+(single)",
                        "AbsoluteErr(s)", "RelativeErr(s)"], floatfmt="3.14f"))

print("\nf(x) = x^3, at x = 1")
dx = np.float64(1)
sx = np.float32(1)
table = []
for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    dDplus = (pow(dx + dDeltaX, 3) - pow(dx,3)) / dDeltaX
    sDplus = (pow(sx + sDeltaX, 3) - pow(sx, 3)) / sDeltaX
    dFprime = np.float64(3)
    sFprime = np.float32(3)
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime
    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError, sRelativeError]
    table.append(row)

print(tabulate(table,
               headers=["D delta x", "f'", "D+(double)", "AbsoluteErr(D)", "RelativeErr(D)", "S Delta x", "D+(single)",
                        "AbsoluteErr(s)", "RelativeErr(s)"], floatfmt="3.14f"))


print("\n--Problem 5--")
# reverse of Fibonacci recurrence
# recurrence f[n+1] = af[n] + f[n-1]
def backward_sequence(a):
    N = [pow(2, i) for i in range(4, 10)]  # N takes different values #2^20
    table = []  # initiate the output table
    for n in N:
        # forward
        f = [1.0, 1.0]  # the starting two terms are 0, 1 for Fibonacci seq
        for j in range(n):
            nextTerm = a * f[-1] + f[-2]  # f[n+1] = af[n] + f[n-1]
            f = [f[-1], nextTerm]  # restructure the storage
            # print(f)

        # backward
        g = [f[-1], f[-2]]
        for k in range(n):
            prevTerm = g[-2] - a * g[-1]  # f[n-1] = f[n+1] - af[n]
            g = [g[-1], prevTerm]  # restructure the storage
            # print(g)

        f_zero = 1.0  # f_0 's value in floating point
        g_zero = g[-1]  # the last term of g is g_0
        AbsErr = abs(f_zero - g_zero)  # absolute error
        RelRrr = AbsErr / f_zero  # relative error
        row = [n, f_zero, g_zero, AbsErr, RelRrr]  # output construction
        table.append(row)

    print(tabulate(table, headers=["N", "Original f_0", "Recomputed f_0", "Absolute Error", "Relative Error"],
                   floatfmt='3.5e'))

if __name__ == '__main__':
    # for a = 1.0
    print("\na = 1.0")
    backward_sequence(1.0)                      # f_{n-1} = f_{n+1} - f_{n}
    print("\na = 1 + \sqrt{2}/10")
    backward_sequence(1.0 + np.sqrt(2) / 10.0)  # f_{n-1} = (1+np.sqrt(2)/10)f_{n+1} - f_{n}


