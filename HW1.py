#  Python 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Source of Error

import numpy as np
import math
from tabulate import tabulate
# Problem 2: two ways of evaluating (x-1)^n

# Evaluate (x-1)^n with direct formula
# Find the relative error to illustrate accuracy when n >= 10, |x-1| <= 1/2
print("--Problem 2--")

n = 10

xValues = [1 + 0.05 * i for i in range(-10,10)]
print(xValues)
table = []
for x in xValues:
    f = pow(x-1,n)

    g = 0
    for k in range(n+1):
        g += math.comb(n,k) * pow(x,n-k) * pow(-1, k)

    try:
        diff = abs(g-f)
        rdiff = diff/g
    except ZeroDivisionError:
        print("Zero Division error")
    row = [x, f, g, diff, rdiff]
    table.append(row)

print(tabulate(table, headers=["x", "f", "g", "Diff", "Relative Diff"], floatfmt='3.13f'))


# Problem 3: Tradeoff between roundoff error and truncation error
print("--Problem 3--")
dDeltaXValues = [np.float64(pow(2,x)) for x in range(-1, -30, -2)] #question: if I use pow then float, do I loss accuracy?
sDeltaXValues = [np.float32(pow(2,x)) for x in range(-1, -30, -2)]
print(type(dDeltaXValues[0]))
print(type(sDeltaXValues[0]))
print("For f(x) = e^x")
table = []
dx = np.float64(1)
sx = np.float32(1)
for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    dDplus = (np.exp(dx + dDeltaX) - np.exp(dx))/dDeltaX
    sDplus = (np.exp(sx + sDeltaX) - np.exp(sx))/sDeltaX

    dFprime = np.e
    sFprime = np.float32(np.e)
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime

    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError,sRelativeError]
    table.append(row)

print(tabulate(table, headers=["D delta x", "f'","D+(double)","Absolute Error (Double)","Relative Error (Double)","S Delta x","D+(single)", "Absolute Error (Single)","Relative Error (Single)"], floatfmt="3.14f"))

print("f(x) = sin(x)")
dx = np.float64(0)
sx = np.float32(0)
table = []
for dDeltaX, sDeltaX in zip(dDeltaXValues, sDeltaXValues):
    dDplus = (np.sin(dx + dDeltaX) - np.sin(dx))/dDeltaX
    sDplus = (np.sin(sx + sDeltaX) - np.sin(sx))/sDeltaX
    dFprime = np.float64(1)
    sFprime = np.float32(1)
    dAbsError = abs(dDplus - dFprime)
    sAbsError = abs(sDplus - sFprime)
    dRelativeError = dAbsError / dFprime
    sRelativeError = sAbsError / sFprime
    row = [dDeltaX, dFprime, dDplus, dAbsError, dRelativeError, sDeltaX, sDplus, sAbsError, sRelativeError]
    table.append(row)

print(tabulate(table, headers=["D delta x", "f'", "D+(double)", "Absolute Error (Double)", "Relative Error (Double)",
                               "S Delta x", "D+(single)", "Absolute Error (Single)", "Relative Error (Single)"],
               floatfmt="3.14f"))

