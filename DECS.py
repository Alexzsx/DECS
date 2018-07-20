from __future__ import division
from numpy import *
import random as rd
from scipy.stats import pearsonr
from dl_simulation import *
from analyze_predictions import *
from run_smaf import smaf
import spams

THREADS = 4

def compare_results(A, B):
    results = list(correlations(A, B, 0))[:-1]
    results += list(compare_distances(A, B))
    results += list(compare_distances(A.T, B.T))
    return results


# Calculate the fitness value
def calFitness_DE(X):
    n = len(X)
    fitness = 0
    for i in range(n):
        fitness += X[i] * X[i]
        #fitness += X[i]**2-10*cos(2*pi*X[i])+10
    return fitness


def calFitness(X, UW):
    n = X.shape[1]
    fitness = np.zeros((1, n))
    for i in range(n):
        fitness[0, i] = 1 - pearsonr(X[:, i], UW[:, i])[0]
    return fitness[0]


def calFitness_1(X, UW):

    return 1 - pearsonr(X, UW)[0]



def mutation(XTemp, F):
    m, n = shape(XTemp)
    XMutationTmp = zeros((m, n))
    for i in range(m):
        r1 = 0
        r2 = 0
        r3 = 0
        while r1 == i or r2 == i or r3 == i or r1 == r2 or r1 == r3 or r2 == r3:
            r1 = rd.randint(0, m - 1)
            r2 = rd.randint(0, m - 1)
            r3 = rd.randint(0, m - 1)

        for j in range(n):
            XMutationTmp[i, j] = XTemp[r1, j] + F * (XTemp[r2, j] - XTemp[r3, j])

    return XMutationTmp


def crossover(XTemp, XMutationTmp, CR):
    m, n = shape(XTemp)
    XCorssOverTmp = zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            r = rd.random()
            if (r <= CR):
                XCorssOverTmp[i, j] = XMutationTmp[i, j]
            else:
                XCorssOverTmp[i, j] = XTemp[i, j]
    return XCorssOverTmp


def selection(XTemp, XCorssOverTmp, fitnessVal, X, U):
    m, n = shape(XTemp)
    fitnessCrossOverVal = zeros(m)
    for i in xrange(m):
        #fitnessCrossOverVal[i, 0] = calFitness(XCorssOverTmp[i])
        fitnessCrossOverVal[i] = calFitness_1(X, U.dot(XCorssOverTmp[i]))
        if (fitnessCrossOverVal[i] < fitnessVal[i]):
            for j in xrange(n):
                XTemp[i, j] = XCorssOverTmp[i, j]
            fitnessVal[i] = fitnessCrossOverVal[i]
    return XTemp, fitnessVal


def crowding(XTemp, XCorssOverTmp, fitnessVal):

    m, n = shape(XTemp)

    for i in xrange(m):
        distance = zeros((m, 1))
        for j in xrange(m):
            distance[j, 0] = linalg.norm(XTemp[j] - XCorssOverTmp[i])
        position = where(distance == min(distance))[0][0]
        fit_XTemp = fitnessVal[position, 0]
        fit_XCorssOverTmp = calFitness(XCorssOverTmp[i])
        if calFitness(XCorssOverTmp[i]) < fit_XTemp:
            #for k in xrange(n):
            #    XTemp[position, k] = XCorssOverTmp[i, k]
            XTemp[position] = XCorssOverTmp[i]
            fitnessVal[position, 0] = fit_XCorssOverTmp
    return XTemp, fitnessVal


def saveBest(fitnessVal, XTemp):
    m = shape(fitnessVal)[0]
    tmp = 0
    for i in xrange(1, m):
        if (fitnessVal[tmp] > fitnessVal[i]):
            tmp = i
    return fitnessVal[tmp][0], XTemp[tmp]
    #print fitnessVal[tmp][0]

if __name__ == "__main__":

    # SMAF setting
    biased_training = 0.
    composition_noise = 0.
    subset_size = 0
    biased_training = 0.
    composition_noise = 0.
    subset_size = 0
    measurements = 200
    sparsity = 10
    dictionary_size = 0.5
    training_dictionary_fraction = 0.05
    SNR = 2.0

    # data load
    #X0 = np.load("./Data/linedata/GSE78779_X0.npy")
    xa = np.load("./Data/linedata/GSE78779_xa.npy")
    xb = np.load("./Data/linedata/GSE78779_xb.npy")

    itr = 0
    while(itr < 1):

        # DE Parameters setting
        NP = 40
        # size = 2
        # xMin = -10
        # xMax = 10
        F = 0.8
        CR = 0.9
        maxItr = 50

        # Initialization
        k = min(int(xa.shape[1] * 3), 150)
        Ws = np.zeros((NP, k, xa.shape[1]))

        UW = (np.random.random((xa.shape[0], k)), np.random.random((k, xa.shape[1])))
        UF, WF = smaf(xa, k, 5, 0.0005, maxItr=10, use_chol=True, activity_lower=0., module_lower=xa.shape[0] / 10, UW=UW,
                     donorm=True, mode=1, mink=3.)

        for i in range(NP):
            lda = np.random.randint(5, 20)
            Ws[i] = sparse_decode(xa, UF, lda, worstFit=1 - 0.0005, mink=3.)

        # Calculate the fitness value
        fitnessVal = zeros((NP, xa.shape[1]))
        for i in range(NP):
            fitnessVal[i] = calFitness(xa, UF.dot(Ws[i]))

        gen = 0
        Xnorm = np.linalg.norm(xa) ** 2 / xa.shape[1]
        while gen <= maxItr:

            for i in range(xa.shape[1]):
                lmd = exp(1 - maxItr / (maxItr + 1 - gen))
                F0 = F * (2 ** lmd)

                Ws_tem = Ws[:, :, i]
                XMutationTmp = mutation(Ws_tem, F)
                XCorssOverTmp = crossover(Ws_tem, XMutationTmp, CR)
                #Ws_tem, fitnessVal[:, i] = selection(Ws_tem, XCorssOverTmp, fitnessVal[:, i], xa[:, i], UF)
				Ws_tem, fitnessVal[:, i] = crowding(Ws_tem, XCorssOverTmp, fitnessVal[:, i])
                WF[:, i] = Ws_tem[np.where(fitnessVal[:, i] == min(fitnessVal[:, i]))[0][0], :]
                Ws[:, :, i] = Ws_tem

            UF = spams.lasso(np.asfortranarray(xa.T), D=np.asfortranarray(WF.T),
                             lambda1=0.0005 * Xnorm, mode=1, numThreads=THREADS, cholesky=True, pos=True)
            UF = np.asarray(UF.todense()).T

            gen += 1

        Results = {}

        x2a, phi, y, w, d, psi = recover_system_knownBasis(xa, measurements, sparsity, Psi=UF, snr=SNR, use_ridge=False)
        Results['DE (training)'] = compare_results(xa, x2a)
        x2b, phi, y, w, d, psi = recover_system_knownBasis(xb, measurements, sparsity, Psi=UF, snr=SNR, use_ridge=False,
                                                           nsr_pool=composition_noise, subset_size=subset_size)
        Results['DE (testing)'] = compare_results(xb, x2b)
        np.save("./Data/heatmap/GSE69405_DECS2.npy", x2b)
        #sys.stdout = open('./Data/linedata/GSE71858_DE_50.log', 'a')
        #print ("run " + str(itr) + ' :')
        # print (data_path, UF.shape, WF.shape, X0.shape, xa.shape, xb.shape)
        for k, v in sorted(Results.items()):
            print '\t'.join([k] + [str(x) for x in v])
        itr += 1
