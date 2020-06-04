#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross Validation functions for domain adaptation classifiers
"""

import numpy as np
from joblib import Parallel, delayed
from itertools import product
from sklearn.metrics import zero_one_loss

#%%

def da_fit_and_score(daClf, X_s, y_s, X_t, y_t,  K, params, idxTuple_s, idxTuple_t, transformer, reverse_cv):

    
    if K is None: 
        X_s_train = X_s[idxTuple_s[0]]
        y_s_train = y_s[idxTuple_s[0]]
        X_s_valid = X_s[idxTuple_s[1]]
        y_s_valid = y_s[idxTuple_s[1]]
        
        X_t_train = X_t[idxTuple_t[0]]
        X_t_valid = X_t[idxTuple_t[1]]
        y_t_valid = y_t[idxTuple_t[1]]
        
#        TRANSFORMER done here
        
        if transformer is not None: 
            transformer.fit(X_s_train)
            # transformer.fit(np.vstack((X_s_train, X_t_train)))
            
            X_s_train = transformer.transform(X_s_train) 
            X_t_train = transformer.transform(X_t_train)
            X_s_valid = transformer.transform(X_s_valid)
            X_t_valid = transformer.transform(X_t_valid)
        
    else:
        if len(X_s.shape) == 2:
            X_s = len(X_s)
            X_t = len(X_t)
        inds_t_train = X_s + idxTuple_t[0]
        inds_t_valid = X_s + idxTuple_t[1] 
        
        projInds = np.hstack((idxTuple_s[0], inds_t_train))
        X_s_train = K[np.ix_(idxTuple_s[0], projInds)]
        y_s_train = y_s[idxTuple_s[0]]
        X_s_valid   = K[np.ix_(idxTuple_s[1], projInds)]
        y_s_valid   = y_s[idxTuple_s[1]]
        
        X_t_train = K[np.ix_(inds_t_train, projInds)]
        X_t_valid   = K[np.ix_(inds_t_valid, projInds)]
        y_t_valid = y_t[idxTuple_t[1]]
            
    daClassifier = daClf(**params)
    
    daClassifier.fit(X_s_train, y_s_train, X_t_train)
    
    # Using the learnt classifier, label the target samples
    dir_score  =  zero_one_loss(y_t_valid, daClassifier.predict(X_t_valid))
    
    
    #Learn a reverse classifier keepig the same parameters: adaptation from target to source, using the pseudo-labels
    if reverse_cv:
        y_t_pseudo = daClf.predict(X_t_train)
        daClassifier.fit(X_t_train, y_t_pseudo, X_s_train)
        rev_score =  zero_one_loss(y_s_valid, daClassifier.predict(X_s_valid))
    else:
        rev_score =  -1
        
    return [rev_score, dir_score]

def DaGridSearchCV(daClf, X_s, y_s, X_t, y_t, K, param_grid, transformer, cv, n_jobs = 1, verbose= 0, reverse_cv = False):
    keyTuple = tuple(param_grid.keys())
    paramValTuple  = tuple(param_grid.values())
    sourceSplits = cv.split(X_s)
    targetSplits = cv.split(X_t)
    parallelizer = Parallel(n_jobs= n_jobs, verbose= verbose)
    return parallelizer(delayed(da_fit_and_score)(daClf, X_s, y_s, X_t, y_t, K, idxTuple_s= idxTuple_s, idxTuple_t = idxTuple_t,
                        transformer= transformer, params= dict(zip(keyTuple, valTuple)), reverse_cv = reverse_cv)
                                for *valTuple, (idxTuple_s, idxTuple_t) in product(*(paramValTuple), zip(sourceSplits, targetSplits)))

def postprocessing(resDict, ind):
    params = resDict["params"]
    nSplits = resDict["nSplits"]
    shape =  tuple(len(v) for v in params.values()) + (nSplits, )
    losses = np.array(resDict["scores_raw"])[:,ind].reshape(shape)
    meanSplitsShape = shape[:-1]
    bestArgs = bestArgsDict(losses, meanSplitsShape, params)
    return bestArgs

def bestArgsDict(losses, shape, params):
    meanLossOverSplits = np.mean(losses, axis= -1) # mean over the different splits
    minLossIndices = np.unravel_index(np.argmin(meanLossOverSplits), shape)
    return dict(zip(params.keys(),[param[k] for param, k in zip(params.values(), minLossIndices)]))