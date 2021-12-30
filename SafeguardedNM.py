#  Problem 4 in Assignent 4
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  optimizer to minimize f(x), with safeguarded Newton's Method

import scipy
import numpy as np
import pdb
# optimizer using modified cholesky
def unsafeguardedNM(f, tol, x0, g, h): #x_0: initial values, in the form of np.array, X: matrix of variables in sympy
    xn = x0 #numpy array
    while True :
        # evaluate gradient and hessian, f(x_n)
        G = g(xn) #  np.array 1*n
        H = h(xn) #  np.array n*n
        if np.linalg.norm(G) < tol:
            break
        #iterative step x_n
        #pdb.set_trace()
        #print(xn)
        if len(x0) >= 2:
            xn = xn - G @ np.linalg.inv(H)
        else:
            xn = xn - G @ 1/H
            print(xn)
    return xn, f(xn)

def f1(x):
    return np.sqrt(x[0]**2 + 1)

# return np.array 2*2
def g1(x):
    return np.array([x[0]**3 + x[0]])
def h1(x):
    return np.array([3*x[0]**2 + 1])

def f2(x, a, b, c, k):
    g = a * x[0]**2 + b * (x[1]-c * np.sin(k * x[0]))**2
    return np.exp(g)

def g2(x, a, b, c, k):
    gprimex0 = 2*a*x[0] + 2*b* (x[1] - np.sin(k * x[0])) * c * k * np.cos(x[0])
    gprimex1 = 2*b * (x[1] - np.sin(k * x[0]))
    return np.array([[gprimex0 * f2(x[0]), gprimex1 * f2(x[1])]])

def h2(x, a, b, c, k):
    pass

x0 = np.array([100])
print(unsafeguardedNM(f1, 0.01, x0, g1, h1))

x0 = np.array([[1, 1]])
print(unsafeguardedNM(f2, 0.01, x0, g2, h2))