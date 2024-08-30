'''
author: Maosheng Yang

We use chebyshev method to compute the concatenation of the higher order laplacians shifts

[Ix Sx S^2x S^3x .... S^kx]
'''

import torch

def chebyshev(L,K,x):
    '''
    x: N x F
    N: dim of simplices
    F: dim of features
    '''
    (N, F) = x.shape
    X = torch.empty(size=(N, F, K),device=x.get_device())
    X[:,:,0] = L@x 
    for k in range(1,K):
        X[:,:,k] = L@X[:,:,k-1]
    
    return X