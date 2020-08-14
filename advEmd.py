 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ot import emd
import cvxpy as cvx
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from scipy.sparse import csr_matrix, vstack, issparse
from proxies import cold
from scipy.special import softmax

#%%
def infEMD(r, c, maxIter, threshold, GammasDotCosts, XXw_s, XXw_t, GammaMats, disp= False):
    i = 0
    error = 1
    newGamma = np.zeros((len(r), len(c)))
    converged= False
    for i in range(maxIter):
        
        p = cvx.Variable(GammasDotCosts.shape[0], nonneg= True)
        problemAlgo = cvx.Problem(cvx.Minimize(cvx.log_sum_exp(cold*GammasDotCosts.T@p)), constraints= [cvx.sum(p)==1])
        
        try:
            problemAlgo.solve(solver= 'MOSEK')
        except cvx.SolverError:
            problemAlgo.solve(solver= 'SCS')
            
        if problemAlgo.status == "infeasible" or p.value is None:
            return None, None, None
        activeVars = p.value
        probas= softmax(cold*np.dot(activeVars, GammasDotCosts))
    
        C_star = makeCostMatAbs(probas, XXw_s, XXw_t)
        val = np.min(np.dot(GammasDotCosts,probas))
        
        
        if C_star is None: return newGamma, error

        
        newGamma = emd(r, c, C_star, numItermax= 10000000)
        newGamma[newGamma < 1e-6] = 0 # sparsification
        newGamma = csr_matrix(newGamma)
        
        error = val - np.sum(newGamma.multiply(C_star))
        if error <= threshold:
            converged= True
            break
        

        if disp:
            print("\n")
            print("iteration = %d" %i)
            print("error = %e" %error)

        activeInds = activeVars >= 1e-6
    
        GammaMats = GammaMats[activeInds,:]
        GammaMats = vstack((GammaMats, csr_matrix(newGamma.reshape((1, len(XXw_s)*len(XXw_t))))))
        GammasDotCosts = GammasDotCosts[activeInds]
        row = makeRowAbs(newGamma, XXw_s, XXw_t, XXw_s.shape[1])
        GammasDotCosts = np.vstack((GammasDotCosts, row))
        activeVars = activeVars[activeInds]
        
    # GammaMats = arguments[-1]
    if not converged: GammaMats = GammaMats[:-1, :]     
    
    activeInds = activeVars>1e-6
    activeVars = activeVars[activeInds]
    activeVars /= np.sum(activeVars)
    GammaMats = GammaMats[activeInds, :]    
    Gamma = GammaMats.T.dot(activeVars).reshape(newGamma.shape)
    Gamma[Gamma < 1e-6] = 0
    
    return csr_matrix(Gamma), error, val

def makeRowAbs(Gamma, XXw_s, XXw_t, d):
    gammas, inds = transportContent(Gamma)
    diffs = gammas[:,None]*(XXw_s[inds[0]] - XXw_t[inds[1]])
    row2 = np.sum(np.abs(diffs), axis= 0)
    return row2

def makeCostMatAbs(probas, XXw_s, XXw_t):
    posInds = probas > 1e-6
    probas = probas[posInds]
    probas /= np.sum(probas)
    return manhattan_distances(probas*XXw_s[:,posInds], probas*XXw_t[:,posInds])    


def transportContent(Gamma):
    if issparse(Gamma):
        indices = Gamma.indices
        indptr  = Gamma.indptr
        inds = np.array([[i,j] for i in range(Gamma.shape[0]) for j in indices[indptr[i]:indptr[i+1]]]).T
        return Gamma.data.copy(), inds
    else:
        logicInds = Gamma>0
        inds = np.where(logicInds)
        return Gamma[logicInds], np.array(inds)