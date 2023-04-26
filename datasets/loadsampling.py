# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:17:59 2017

@author: jlc516
"""
import numpy as np
import numpy.linalg as linalg
from scipy.stats import norm

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def sampleruniform(loads):
    """ input is a list of loads
        output is a matrix of load sets
        the load are sampled with +-50% variation of each single load
    """
    matrixmulti = np.ones( shape = (len(loads), 2*len(loads)+1))
    for i in range(len(loads)*2+1):
        if i>=1 and i%2 != 0:
            matrixmulti[int(i/2),i] = 1.5
        elif i>=1 and i%2 == 0:
            matrixmulti[int((i-1)/2),i] = 0.5
    loadsm = np.multiply(np.repeat(loads[:,np.newaxis], 2*len(loads)+1, 1),matrixmulti) 
    return loadsm

def samplersteps(loads,sampletheloads,steps):
    """ input is a list of loads
        output is a matrix of load sets
        all loads are sampled in steps 0:0.2:2
    """
    numsamples = pow(len(steps),len(sampletheloads))
    #combi = pow(len(sampletheloads), 2)
    matrixmulti = np.ones( shape = (len(loads), numsamples ) )
    for j in range(len(sampletheloads)):
        makestepat = pow(len(steps),len(sampletheloads)-j-1)
        counter = 0
        currentstep = 0
        for i in range(numsamples):
            matrixmulti[sampletheloads[j],i] = steps[currentstep]
            counter = counter +1
            if counter >= makestepat:
                counter = 0
                currentstep = currentstep + 1
                if currentstep > len(steps)-1:
                    currentstep = 0

    loadsm = np.multiply(np.repeat(loads[:,np.newaxis], numsamples, 1),matrixmulti)
    return  loadsm

def samplermontecarlo(LB, UB, numbersamples):
    #tic1 = time.time()
    
    UBLB = UB-LB
    
    if np.size(LB)==1:
        MLB=np.repeat(LB,numbersamples)
        MUBLB = np.repeat(UBLB, numbersamples)
    else:
        MLB = np.repeat(LB[:,np.newaxis], numbersamples, 1) 
        MUBLB = np.repeat(UBLB[:,np.newaxis], numbersamples, 1)

    MCM = MLB + np.multiply(np.random.rand(np.size(UB),numbersamples),MUBLB)

    return MCM

def samplermontecarlo_normal(MU, SIG, numbersamples):
    #tic1 = time.time()
    
    
    if np.size(MU)==1:
        MMU = np.repeat(MU, numbersamples)
        MSIG = np.repeat(SIG, numbersamples)
    else:
        MMU = np.repeat(MU[:,np.newaxis], numbersamples, 1)
        MSIG = np.repeat(SIG[:,np.newaxis], numbersamples, 1)

    MCM = np.random.normal(loc = MMU, scale = MSIG, size = (np.size(MU),numbersamples))

    return MCM
    
def kumaraswamymontecarlo(a, b, c, LB, UB, num_samples):

    num_variables = len(LB)

    MLB = np.repeat(LB[:,np.newaxis], num_samples, 1) 
    UBLB = UB-LB
    MUBLB = np.repeat(UBLB[:,np.newaxis], num_samples, 1)

    uncorrelated = np.random.standard_normal((num_variables, num_samples))
    
    cov = c * np.ones(shape = (num_variables,num_variables)) + (1-c)*np.identity(num_variables)    
    L = linalg.cholesky(cov)

    correlated = np.dot(L, uncorrelated)
    cdf_correlated = norm.cdf(correlated)
    
    karamsy = pow((1-pow((1-cdf_correlated),(1/b))),(1/a))
    
    #probabilities = a* *
    
    MCM = MLB + np.multiply(karamsy,MUBLB)
    
    return MCM                
   
def beta(a, b, num_samples):
     
     samples = np.random.beta(a,b,size=num_samples)
     
     return samples
 