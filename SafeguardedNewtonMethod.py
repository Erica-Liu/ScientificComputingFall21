#  Problem 4 in Assignment 4
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  optimizer to minimize f(x), with safeguarded Newton's Method

import scipy
import numpy as np
import pdb
from sympy import symbols, Matrix, Function, simplify, sqrt, exp, sin, hessian, solve, init_printing
from scipy.linalg import ldl
import matplotlib.pyplot as plt
init_printing()

#determine the number of variables

# optimizer using only modified cholesky
def MCsafeguardedNM(f, tol, X, x_0): #x_0: initial values, in the form of np.array, X: matrix of variables in sympy
    gradf = simplify(f.jacobian(X))
    hessianf = simplify(hessian(f, X))
    x_n = x_0
    f_eval = 0
    N = 10 ** 3
    iteration = 1
    while iteration <= N :
        # evaluate gradient and hessian, f(x_n)
        elements = {x:xval for x,xval in zip(X, x_n[0])}
        #pdb.set_trace()
        f_eval = f.evalf(subs=elements)
        G_sym = gradf.evalf(subs=elements)
        H_sym = hessianf.evalf(subs=elements)
        G = np.array(G_sym).astype('float')
        H = np.array(H_sym).astype('float')
        lu, di, perm = ldl(H)
        #pdb.set_trace()
        H_tilde = lu @ np.absolute(di) @ lu.T
        if np.linalg.norm(G) < tol:
            print(np.linalg.norm(G))
            break
        #iterative step x_n
        print(x_n)
        x_n = x_n - (G @ np.linalg.inv(H_tilde))
        iteration += 1
    return x_n, f_eval

# optimizer using newton's method
def unsafeguardedNM(f, tol, X, x_0, N = 10 ** 3): #x_0: initial values, in the form of np.array, X: matrix of variables in sympy
    print("\n===========Unsafeguarded starts===========")
    gradf = simplify(f.jacobian(X))
    hessianf = simplify(hessian(f, X))
    x_n = x_0
    f_eval = 0
    N = 10 ** 3
    iteration = 1
    while iteration <= N :
        # evaluate gradient and hessian, f(x_n)
        elements = {x:xval for x,xval in zip(X, x_n[0])}
        # pdb.set_trace()
        f_eval = f.evalf(subs=elements)
        G_sym = gradf.evalf(subs=elements)
        H_sym = hessianf.evalf(subs=elements)
        G = np.array(G_sym).astype('float')
        H = np.array(H_sym).astype('float')
        if np.linalg.norm(G) < tol:
            print(np.linalg.norm(G))
            break
        #iterative step x_n
        print("x_{0} : {1}".format(iteration, x_n[0]))
        try:
            x_n = x_n - (G @ np.linalg.inv(H))
        except:
            print("unsafeguarded not applicable")
            break
        iteration += 1
    return x_n, f_eval

global N


# optimizer using linear search + modified cholesky
def safeguardedNM(f: Matrix, tol: float, X: Matrix, x_0: list , N: int = 10 ** 3) -> Matrix: #x_0: initial values, in the form of np.array, X: matrix of variables in sympy
    print("\n==========Safeguarded starts:==========")
    gradf = simplify(f.jacobian(X))
    hessianf = simplify(hessian(f, X))
    x_n = x_0
    f_eval = 0
    iteration = 1
    singular = False
    while iteration <= N :
        #pdb.set_trace()
        # evaluate gradient and hessian, f(x_n)
        elements = {x:xval for x,xval in zip(X, x_n[0])}
        f_eval = f.evalf(subs=elements)
        G_sym = gradf.evalf(subs=elements)
        H_sym = hessianf.evalf(subs=elements)
        G = np.array(G_sym).astype('float')
        H = np.array(H_sym).astype('float')
        lu, di, perm = ldl(H)
        #pdb.set_trace()
        H_tilde = lu @ np.absolute(di) @ lu.T
        if np.linalg.norm(G) < tol:
            print(np.linalg.norm(G))
            break
        # linear search
        s = 2
        print("x_{0} : {1}".format(iteration, x_n[0]))
        x_temp = x_n
        f_temp_eval = f_eval
        while f_temp_eval[0] >= f_eval[0]:
            s = s/2
            try:
                x_temp = x_n - s * (G @ np.linalg.inv(H_tilde))
            except:
                print("safeguarded not applicable")
                singular = True
                break
            elements = {x: xval for x, xval in zip(X, x_temp[0])}
            f_temp_eval = f.evalf(subs=elements)
        #iterative step x_n
        x_n = x_temp
        iteration += 1
        if singular:
            break
    return x_n, f_temp_eval

def testfunc_1():
    # create functions
    x= symbols('x')
    f= symbols('f', cls=Function)

    X = Matrix([x])
    f = Matrix([sqrt(x**2 + 1)])
    x_0 = np.array([[2]])
    print("test function:{f}".format(f=f[0]))
    x_n, f_eval = unsafeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))
    x_n, f_eval = safeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))
def testfunc_2():
    x, y = symbols('x y')
    f, g, h = symbols('f g h', cls=Function)
    # example with 2 variables
    x, y = symbols('x y')
    f, g, h = symbols('f g h', cls=Function)
    a = 1
    b = 5
    c = 1
    k = 5
    X = Matrix([x,y])
    f = Matrix([exp(a*x**2 + b*(y - c*sin(k*x))**2)])

    x_0 = np.array([[2,2]])
    print("test function:{f}".format(f=f[0]))
    x_n, f_eval = unsafeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))
    x_n, f_eval = safeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))

def testfunc_3():
    # function test routine
    # create variables
    a = 3
    n = 5
    X = []
    x_0 = [1, 2.3, 4, 5, 2]
    for i in range(1,n+1):
        # variable name
        x_name = 'x' + str(i)
        X.append(symbols(x_name))

    #create function f
    x_0 = np.array([x_0])
    f = 0
    for k in range(n-1):
        f += X[k] + (X[k+1] - X[k])**2 + a * (X[k+1] - X[k])**4
    f += X[-1]
    f += X[0]**2 + a * X[0]**4 + X[-1]**2 + a * X[-1]**4
    f = Matrix([f])
    X = Matrix(X)
    #pdb.set_trace()
    print("test function:{f}".format(f=f[0]))
    x_n, f_eval = unsafeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))
    x_n, f_eval = safeguardedNM(f, 0.01, X, x_0)
    print("x_n : {x_n}, optimum value : {f_eval}".format(x_n=x_n[0], f_eval=f_eval[0]))

    # plot x_k of k
    plt.plot([i for i in range(1,n+1)], x_n[0])
    plt.xlabel("k")
    plt.ylabel("x_k")
    plt.title("Test 3 w/a=3, n=5 : x_k vs k")
    plt.show()


def contour_plot():
    a = 1
    b = 5
    c = 1
    k = 5

    fig, ax = plt.subplots()
    x = np.arange(-2, 2, 0.01)
    y = x.reshape(-1, 1)
    h = np.exp(a * x**2 + b*(y - c*np.sin(k * x))**2)
    cs = plt.contourf(h, levels=[1,1.1,1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 4, 8, 16, 32, 64, 100, 1000, 10000, 100000],
        colors=[ '#930000', '#0000E3', '#AE0000', '#2828FF', '#CE0000', '#4A4AFF', '#EA0000', '#6A6AFF',
                 '#FF0000', '#7D7DFF', '#FF2D2D', '#9393FF', '#FF5151', '#AAAAFF', '#ff7575', '#B9B9FF', '#FF9797', '#CECEFF',
                 '#FFB5B5', '#DDDDFF', '#FFD2D2', '#ECECFF', '#FFECEC','#FBFBFF'], extend='both')
    cs.cmap.set_over('#FFECEC')
    cs.cmap.set_under('red')
    cs.changed()
    plt.xticks(np.arange(0,400,50), np.arange(-2,2,0.5) )
    plt.yticks(np.arange(0, 400, 50), np.arange(-2, 2, 0.5))
    plt.title("Coutour Plot Test 2 w/ a={a}, b={b}, c={c}, k={k}".format(a=a, b=b, c=c, k=k))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
    #testfunc_1()

    #testfunc_2()
    #contour_plot()

    testfunc_3()
