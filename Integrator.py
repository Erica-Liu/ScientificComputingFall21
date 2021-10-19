#  Python 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Local Analysis

# Rectangle rule first order fixed value
def RR1f(a, b, n, m):
    h = (b-a)/n             # calculate the step size
    x = a                   # starting point
    I = 0.0                 # global integration
    for i in range(n):
        I += h * m.f(x)
        x += h
    return I

# mid-point rule second order fixed value
def MR2f(a, b, n, m):
    h = (b-a)/n
    x = a + h/2
    I = 0.0
    for i in range(n):
        I += h * m.f(x)
        x += h
    return I

# Simpson's rule 6th order fixed value
def SR4f(a, b, n, m):
    h = (b-a)/n
    x = a
    I = 0.0
    for i in range(n):
        I += (h/6) * (m.f(x) + 4 * m.f(x + h/2) + m.f(x + h))
        x += h
    return I

# adaptive algorithm
# the best approximation, the number of points needed to get it, and character string
# that tells you whether the requested error was achieved or not
# N_max
def RR1a(a, b, n0, m, tol):
    A = RR1f(a, b, n0, m)
    A2 = RR1f(a, b, 2*n0, m)
    N_max = 2**23
    flag = True
    while abs(A - A2) > tol and n0 < N_max: #first order
        n0 = 2*n0
        A = A2
        A2 = RR1f(a, b, 2*n0, m)
    if n0 >= N_max:
        flag = False
        A2 = RR1f(a, b, n0, m)
    return A2, n0, flag

# 4th order Simpson Method
def SR4a(a, b, n0, m, tol):
    A = SR4f(a, b, n0, m)
    A2 = SR4f(a, b, 2*n0, m)
    N_max = 2**23
    flag = True
    while abs((A - A2)/A)> tol and n0 < N_max: #first order
        n0 = 2*n0
        A = A2
        A2 = SR4f(a, b, 2*n0, m)
    if n0 >= N_max:
        flag = False
        A2 = SR4f(a, b, n0, m)
    return A2, n0, flag


