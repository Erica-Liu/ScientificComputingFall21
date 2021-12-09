#  Problem 4 in Assignent 4
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  optimizer to minimize f(x), with safeguarded Newton's Method

import scipy
from sympy import symbols, Matrix, Function, simplify, exp, sin, hessian, solve, init_printing

init_printing()
# create functions
x, y = symbols('x y')
f, g, h = symbols('f g h', cls=Function)
a = 1
b = 1
k = 1
X = Matrix([x])
f = Matrix([(x**2 + 1)**(-1/2)])
gradf = simplify(f.jacobian(X))

print(gradf)
"""
X = Matrix([x,y])
f = Matrix([exp(a*x**2 + b*(y - c*sin(k*x))**2)])

h = 2*x1 + 3*x2
g =  x1**2 + x2**2 - 10



# optimizer using modified cholesky
def safeguardedNM(f, tol, x_this):
    gradf = simplify(f.jacobian(X))
    hessianf = simplify(hessian(f, X))
    while f(x0) :

        x_next = x_this -




"""
# test

