#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ot.lp import emd
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from advEmd import transportContent
from scipy.sparse import csr_matrix, issparse
from scipy.optimize import minimize
import proxies as prox
from scipy.special import softmax, logsumexp
import advEmd as advEmd

class Madaot():
    
    def __init__(self, delta= 1, zeta= 1e-5, thd1= 1e-4, thd2= 1e-7, nIter= 10):
        self.delta = delta
        self.zeta = zeta 
        self.thd1 = thd1
        self.thd2 = thd2
        self.coefs = None
        self.log = None
        self.nIter = nIter
        
    def fit(self, X_s, y_s, X_t):
        self.coefs, self.log, self.transport = learnClassifAdaptGrad(X_s, y_s, X_t, self.delta, self.zeta, w_sol= self.coefs, 
                              Gamma_sol= None, thd1= self.thd1, thd2= self.thd2, nIter= self.nIter)
        return self
        
    def decisionFunc(self, X):
        return np.dot(X, self.coefs)
    
    def predict(self, X):
        return np.sign(self.decisionFunc(X))
    


def objFunSmooth( w, X_s, y_s, X_t, zeta, delta, gammas, inds, class_weights):
    
    src, gradSrc = sourceHingeSmooth( w, X_s, y_s, class_weights)    
    discVec, gradDiscVec= alignCorrespAbsSmooth( w, X_s[inds[0]].T, X_t[inds[1]].T, gammas)
    return src + delta*discVec + zeta*np.dot(w, w) , gradSrc + delta*gradDiscVec + zeta*2*w



"""
    Finds the classifier minimizing the objective function
"""     
def learnClassifAdaptGrad(X_s, y_s, X_t, delta, zeta, w_sol= None, Gamma_sol= None, thd1= 1e-4, thd2= 1e-7, nIter= 10):
   
    class_weights, _ = classWeights(y_s)
        
    d = X_s.shape[1]
    if w_sol is None: w_sol = np.random.uniform(-1,1,d)
            
    n = len(X_t)

    
    # Intializing the algorithm
    
    print("[", end= '')
    if Gamma_sol is None:
        Gamma_sol = emd(class_weights, np.ones(n)/n, cdist(X_s, X_t, metric= 'sqeuclidean'), numItermax= 1000000)
        Gamma_sol[Gamma_sol<1e-6]= 0
    print("-", end= "")
    if not issparse(Gamma_sol):
        Gamma_sol = csr_matrix(Gamma_sol)
    gammas, inds = transportContent(Gamma_sol)
    
    
    vecMinArgs = {"jac": True, 
                    "options": {"disp": False, "gtol": 1e-4, "maxiter": 100, "ftol": 1e-6}, 
                  "method": 'L-BFGS-B'
                  }
        
    errs = []
    w_diffs = []
    grad_w_norms = []
    Gamma_diffs = []
    value_old= 1e16
    values = [value_old]
    w_old = w_sol.copy()
    Gamma_old = Gamma_sol.copy()
    
    for i in range(nIter):
        
        # Optimize over  w: start from previous solution
        
        objFunArgs = (X_s, y_s,  X_t, zeta, delta, gammas, inds, class_weights)
        
        res = minimize(objFunSmooth, x0= w_sol, args= objFunArgs, **vecMinArgs)
        w_sol = res['x']
        grad_w_sol = res['jac']
        value = res['fun']
        
        if res['success']: print(">", end= '')
        else: print("i", end= '')
        
        w_diffs.append(cosine(w_sol, w_old))
        grad_w_norms.append(np.max(np.abs(grad_w_sol)))
        
        err = (value_old - value)/max(value, value_old, 1)
        errs.append(err)
        values.append(value)
        
        if err < -thd1: print('x', end= '')
        if abs(err)<= thd1 or w_diffs[-1] < thd2:
            break

        #  Optimize over Gamma: Compute the minimax transport plan, starting from previous Gamma
        Gamma_sol, gammas, inds, val = minimaxOtTermSmooth(w_sol, X_s, X_t,  class_weights, 
                                                              Gamma_sol, maxIter= 100, threshold= 1e-4, verbose= False)
        
        Gamma_diffs.append(np.max(np.abs(Gamma_sol - Gamma_old)))
        print("-", end= '')

       
        
        # update the old values and solutions0
        w_old     = w_sol.copy()
        Gamma_old     = Gamma_sol.copy()
        value_old = value
    
    print("]") # end of optimization
    dictionary = {"values": values, 
                  "errors": errs, 
                  "w_diffs": w_diffs, 
                  "Gamma_diffs": Gamma_diffs
                  }

    return w_sol, dictionary, Gamma_sol

def sourceHingeSmooth(w, X_s, y_s, class_weights):
    yXa = y_s*np.dot(X_s, w)
    val = np.dot(class_weights, prox.smPos(1 - yXa))
    signedMarginViol = -y_s*prox.derSmPos(1 - yXa)
    grad = np.dot(class_weights*signedMarginViol, X_s)
    return val, grad

def alignCorrespAbsSmooth(w, X_s_T, X_t_T, gammas):
    
    DxxA = X_s_T*np.dot(w, X_s_T) - X_t_T*np.dot(w, X_t_T)
    meanAbsDxxA = np.dot(prox.smAbs(DxxA), gammas)
    softMaxInd = softmax(prox.cold*meanAbsDxxA)
    signDxxA = prox.derSmAbs(DxxA)
    jacPerPair = X_s_T*np.dot(softMaxInd, X_s_T*signDxxA) - X_t_T*np.dot(softMaxInd, X_t_T*signDxxA)
    
    return prox.temper*logsumexp(prox.cold*meanAbsDxxA), np.dot(jacPerPair, gammas)

def minimaxOtTermSmooth(w, X_s, X_t, class_weights, W0, maxIter, threshold, verbose= False):
    n = len(X_t)
    m = len(X_s)
    XXw_s = np.dot(X_s, w)[:,None]*X_s
    XXw_t = np.dot(X_t, w)[:,None]*X_t
    GammasDotCosts = np.array([advEmd.makeRowAbs(W0, XXw_s, XXw_t, XXw_s.shape[1])])
    Gamma, minimaxErr, value = advEmd.infEMD(class_weights, np.ones(n)/n, maxIter, threshold, GammasDotCosts, XXw_s, XXw_t, csr_matrix(W0.reshape(1,m*n)))
    gammas, inds = advEmd.transportContent(Gamma)
    return Gamma, gammas, inds, value

def classWeights(y):
    classes = np.unique(y)
    binaryClasses = (y == classes[:,None])
    classCounts = np.sum(binaryClasses, axis= 1)
    weights = np.sum(binaryClasses/classCounts[:,None], axis= 0)/len(classes)
    return weights/np.sum(weights), 2*binaryClasses-1