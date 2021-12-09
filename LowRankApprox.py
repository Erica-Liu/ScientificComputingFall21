#  Problem 3 in Homework 3
#  Scientific Computing, Fall 2021, yl8801@nyu.edu
#  Least Square Fitting

import numpy   as np      # general numpy
import time
import numpy.linalg as la

from tabulate import tabulate

def readData(SourcePointsFile, TargetPointsFile):
    #   Source points
    inFile       = open( SourcePointsFile, "r")  # open the source points file
    firstLine    = inFile.readline()             # the first line has the number of points
    nPoints      = int(firstLine)                # convert the number from string to int
    sourcePoints = np.zeros([nPoints,3])         # the source points array
    for p in range(nPoints):
       dataLine = inFile.readline()              # there is one point per line
       words = dataLine.split()                  # each word is a number
       x     = np.float64(words[0])              # x, y, and z coordinates
       y     = np.float64(words[1])              # convert from string to float
       z     = np.float64(words[2])
       sourcePoints[p,0] = x                     # save the numbers in the numpy array
       sourcePoints[p,1] = y
       sourcePoints[p,2] = z
    inFile.close()
    #   target points
    inFile       = open( TargetPointsFile, "r")  # open the source points file
    firstLine    = inFile.readline()             # the first line has the number of points
    nPoints      = int(firstLine)                # convert the number from string to int
    targetPoints = np.zeros([nPoints,3])         # the source points array
    for p in range(nPoints):
       dataLine = inFile.readline()              # there is one point per line
       words = dataLine.split()                  # each word is a number
       x     = np.float64(words[0])              # x, y, and z coordinates
       y     = np.float64(words[1])              # convert from string to float
       z     = np.float64(words[2])
       targetPoints[p,0] = x                     # save the numbers in the numpy array
       targetPoints[p,1] = y
       targetPoints[p,2] = z
    inFile.close()
    return sourcePoints, targetPoints


# generate matrix A
def genMatrix(sourcePoints, targetPoints):
    n_s = sourcePoints.shape[0]
    n_t = targetPoints.shape[0]
    A = np.zeros((n_t, n_s))
    for j in range(n_t): # target j 3000
        for i in range(n_s): # source i 5000
            A[j,i] = 1/np.sum(np.square(sourcePoints[i]-targetPoints[j]))
    return A

def svd(A):
    u, sigma, vh = la.svd(A)
    return u, sigma, vh

def lowRankApproximate(u, sigma, vh, tol):
    # find the smallest rank
    k = 1
    while(sigma[k] > tol):
        k += 1
    sigma_k = np.diag(sigma[:k])
    vh_k = vh[:k, :]
    u_k = u[:,:k]
    A_k = u_k @ sigma_k @ vh_k
    return sigma_k @ vh_k, u_k, k

def lowRankProduct(sigma_kvh_k, u_k, w):
    temp = sigma_kvh_k @ w
    return u_k @ temp

def matrixVectorProduct(A, w):
    return A @ w

def total_illumination(b):
    return np.sum(b)

def gen_r(z_low, z_high):
    return np.random.uniform(z_low, z_high, 3)

def illumination_full(r_list, A):
    # generate w list
    w_list = []
    for r in r_list:
        n_s = sourcePoints.shape[0]
        w = np.zeros([n_s, 1])
        for i in range(n_s):
            w[i, 0] = 1 / np.sum(np.square(sourcePoints[i] - r))
        w_list.append(w)
    I = 0
    for w in w_list:
        b = A @ w
        I += np.sum(b)
    return I

def illumination_low_rank(r_list, A, tolerence=.01):
    # generate w list
    w_list = []
    for r in r_list:
        n_s = sourcePoints.shape[0]
        w = np.zeros([n_s, 1])
        for i in range(n_s):
            w[i, 0] = 1 / np.sum(np.square(sourcePoints[i] - r))
        w_list.append(w)

    #SVD
    u, sigma, vh = svd(A)
    # generating \Sigma_kV_k^t, and U_k
    sigma_kvh_k, u_k, k = lowRankApproximate(u, sigma, vh, tolerence)
    I = 0
    for w in w_list:
        b_k = u_k @ (sigma_kvh_k @ w)
        I += np.sum(b_k)
    return I


if __name__ == '__main__':
    SourcePointsFile = "SourcePoints.txt"
    TargetPointsFile = "TargetPoints.txt"

    # reading data
    startT = time.time()
    sourcePoints, targetPoints = readData(SourcePointsFile, TargetPointsFile)
    readDataT = time.time()
    Info = "{phase:25} time : {time:14.15e}"
    Info = Info.format(phase="Reading data",time=(readDataT - startT))
    print(Info)

    # generate A
    startT = time.time()
    A = genMatrix(sourcePoints, targetPoints)
    genAT = time.time()
    Info = "{phase:25} time : {time:14.15e}"
    Info = Info.format(phase = "Generating A", time=(genAT - startT))
    print(Info)

    # choose r randomly
    num_r_list = [500*k for k in range(1,10)]
    table = []
    for num_r in num_r_list:
        z_low, z_high = -5, 5
        r_list = []
        for i in range(num_r):
            r_list.append(gen_r(z_low, z_high))

        fullStartT = time.time()
        illumination_full(r_list, A)
        fullEndT = time.time()
        full_time = fullEndT - fullStartT

        lowStartT = time.time()
        illumination_low_rank(r_list, A)
        lowEndT = time.time()
        low_time = lowEndT - lowStartT

        diff = full_time - low_time
        table.append([num_r, full_time, low_time, diff])
    print(tabulate(table, headers=["num r", "full time", "low rank time", "Diff"], floatfmt='4.6f'))


        # generate vector_list w
        w_list = []
        for r in r_list:
            n_s = sourcePoints.shape[0]
            w = np.zeros([n_s, 1])
            for i in range(n_s):
                w[i, 0] = 1 / np.sum(np.square(sourcePoints[i] - r))
            w_list.append(w)

        print("=====Full rank multiplication method======")
        # direct multiplication
        startT = time.time()
        I = 0
        for w in w_list:
            b = A @ w
            I += total_illumination(b)
        directT = time.time()
        Info = "{phase:25} time : {time:14.15e}"
        Info = Info.format(phase="Direct method total", time = (directT - startT))
        print(Info)

        print("=====low rank approximation method======")
        # low rank multiplication
        lowRankMulStartT = time.time()
        # SVD
        startT = time.time()
        u, sigma, vh = svd(A)
        svdAT = time.time()
        Info = "{phase:25} time : {time:14.15e}"
        Info = Info.format(phase="SVD of A", time=(svdAT - startT))
        print(Info)

        # generating \Sigma_kV_k^t, and U_k
        tolerence = .01
        startT = time.time()
        sigma_kvh_k, u_k, k = lowRankApproximate(u, sigma, vh, tolerence)
        genKApproxT = time.time()
        Info = "{phase:25} time : {time:14.15e}, {rank:4}"
        Info = Info.format(rank=k, phase="Low rank approximation", time=(genKApproxT - startT))
        print(Info)

        I = 0
        for w in w_list:
            b_k_temp = sigma_kvh_k @ w
            b_k = u_k @ b_k_temp
            I += total_illumination(b_k)

        lowRankMulEndT = time.time()
        Info = "{phase:25} time : {time:14.15e}"
        Info = Info.format(phase = "low rank method total", time = (lowRankMulEndT - lowRankMulStartT))
        print(Info)

        #accuracy
        print("accuracy of low rank approximation")
        #compare (tol||w||) with ||bk-b||
        for w in w_list[:10]:
            b = A @ w
            b_k = u_k @ (sigma_kvh_k @ w)
            predicted_acc = tolerence * la.norm(w)
            actural_acc = la.norm(b_k - b)
            Info = "{predicted_acc:14.15e}  vs  {actural_acc:14.15e}"
            Info = Info.format(predicted_acc=predicted_acc, actural_acc=actural_acc)
            print(Info)










