#  Demonstration Python module for Week 3
#  Scientific Computing, Fall 2021, goodman@cims.nyu.edu
#  Compute the exponential of a matrix three ways

import numpy        as np  # general numpy
import numpy.linalg as la  # linear algebra routines

def conditionNumber(M):
    return la.norm(M)*la.norm(la.inv(M))

def mee(L, t):  # compute the matrix exponential using eigenvalues
    [d, d] = L.shape
    lam, R = la.eig(L)  # lam = eigenvalues, R = right eigenvectors
    d_elamt = np.zeros([d, d])  # diagonal matrix with e^{-lambda_j t}
    for j in range(d):
        d_elamt[j, j] = np.exp(lam[j] * t)

    # add condition number of R for this one
    conN = conditionNumber(R)
    return (R @ d_elamt @ la.inv(R), conN)


def med(L, t, k):  # compute the matrix exponential, differential equation
    [d, d] = L.shape
    n = 2 ** k  # number of intervals
    dt = t / n  # time step
    I = np.identity(d)  # to make the next formula simpler

    #    S(dt) computed using 4 Taylor series terms in Horner's rule form

    S = I + dt * L @ (I + (1 / 2) * dt * L @ (I + (1 / 3) * dt * L @ (I + (1 / 4) * dt * L)))

    #   Raise this to the power n using repeated doublings S_j = S^{2^j} until j=k

    for j in range(k):
        S = S @ S  # S^{2^{j-1}} becomes S^{2^j} when you square it
    return (S)

# print out the largest norm of items
def meT(L, t, n):  # compute the matrix exponential using Taylor series
    # record the norm of (t^k/k!)L^k
    term_norms = []
    [d, d] = L.shape
    tLk = np.identity(d)  # will be (tL)^k
    kf = 1  # will be k! = k factorial
    S = np.identity(d)  # will be the answer = sum (1/k!)((tL)^k
    for k in range(1, n):
        kf = k * kf  # multiply by k to turn (k-1)! into k!
        tLk = t * ((tLk) @ L)  # turn (k-1)-th power into k-th power(tL)^k,
        S += (1 / kf) * tLk  # (1/k!)(tL)^k is the k-th Taylor series term
        term_norms.append(la.norm((1 / kf) * tLk))
    return (S, max(term_norms))


#    Physical parameters

d = 2  # size of the matrix
r_u = .5  # rate of going up:   k -: k+1 transition
r_d = .2 # rate of going down: k -> k-1 transition
r_l = (r_u + r_d)  # "loss rate" = rate to leave state k
t = 1.  # time: Compute exp(tL)

#     Computational parameters

n_T = 200  # number of Taylor series terms
k = 11  # n = 2^k intervals for Runge Kutta, dt = t/n

#    The generator matrix L, with up rate on the super-diagonal
#    and down rate on the sub-diagonal.

L = np.zeros([d, d])
for i in range(1, d):
    L[i, i - 1] = r_d
    L[i - 1, i] = r_u

L[0, 0] = - r_u
L[d - 1, d - 1] = - r_d
for i in range(1, d - 1):
    L[i, i] = -r_l

#    Compute S = exp(tL) three ways

Se, condNum = mee(L, t)
ST, maxNorm = meT(L, t, n_T)
Sd = med(L, t, k)

#     The RMS differences between the computed matrices

rms_eT = np.sqrt(np.sum((Se - ST) ** 2))
rms_ed = np.sqrt(np.sum((Se - Sd) ** 2))
rms_Td = np.sqrt(np.sum((ST - Sd) ** 2))

#     Formatted output

print("\nRMS differences between computed matrix exponentials\n")
runInfo = "up rate = {r_u:8.1f}, down rate = {r_d:8.1f}, dimension is {d:3d}\n"
runInfo = runInfo.format(r_u=r_u, r_d=r_d, d=d)
print(runInfo)

#print out the information for maxnorm and
mee_info = "Eigenvalue method the condition number of R: {cond_num:14.6e}"
mee_info = mee_info.format(cond_num = condNum)
print(mee_info)

meT_info = "Taylor series method the largest norm:       {max_norm:14.6e}"
meT_info = meT_info.format(max_norm = maxNorm)
print(meT_info)

eT_info = "\neigenvalue  vs. Taylor series: {rms_eT:14.6e},  with {n:4d} terms"
ed_info = "eigenvalue  vs. RungeKutta:    {rms_ed:14.6e},  with   k = {k:2d}"
Td_info = "Runge Kutta vs. Taylor series: {rms_Td:14.6e}"
eT_info = eT_info.format(rms_eT=rms_eT, n=n_T)
ed_info = ed_info.format(rms_ed=rms_ed, k=k)
Td_info = Td_info.format(rms_Td=rms_Td)

print(eT_info)
print(ed_info)
print(Td_info)