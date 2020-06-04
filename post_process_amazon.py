#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sys import path as path
path.append("../")
from os import path, makedirs
from sklearn.datasets import load_svmlight_file

# resultsPath = "results/DA/"

# if not path.exists(resultsPath):
#     makedirs(resultsPath)
    
from itertools import permutations
from scipy.sparse import csr_matrix, vstack
from madaot import Madaot
from sklearn.metrics import accuracy_score

#%%

domainPermutations = list(permutations((
                                    "books", 
                                    "dvd", 
                                    "electronics",
                                    "kitchen", 
                                     ), 2))

sourceAccuracy = dict([("%s->%s"%(source,target), 0) for source, target in domainPermutations])
targetAccuracy = dict([("%s->%s"%(source,target), 0) for source, target in domainPermutations])
testAccuracy   = dict([("%s->%s"%(source,target), 0) for source, target in domainPermutations])

for source, target in domainPermutations: 
    print("\n%s -> %s"%(source, target))

    X_s, y_s = load_svmlight_file("../data/amazon_tfidf_svmlight/svmlight/%s.%s_source.svmlight"%(source, target))  
    X_s = X_s.toarray()

    X_t, y_t = load_svmlight_file("../data/amazon_tfidf_svmlight/svmlight/%s.%s_target.svmlight"%(source, target))
    X_t = X_t.toarray()
    
    d_s = X_s.shape[1]
    d_t = X_t.shape[1]
    if d_s<d_t:
        dim = d_t
        X_s = np.hstack((X_s, np.zeros((X_s.shape[0], d_t-d_s))))
    else:
        dim = d_s
        X_t = np.hstack((X_t, np.zeros((X_t.shape[0], d_s-d_t))))
    
    X_s = csr_matrix(X_s)
    X_t = csr_matrix(X_t)
    X_test, y_test = load_svmlight_file("../data/amazon_tfidf_svmlight/svmlight/%s.%s_test.svmlight"%(source, target))
    d_test = X_test.shape[1]
    if d_test<dim:
        X_test = np.hstack(X_test, np.zeros((len(X_test), dim - d_test)))
    else:
        X_test = X_test[:,:dim]
        
        
    projSpace = vstack((X_s, X_t))
    K = projSpace.dot(projSpace.T).toarray()
    
    X_s_proj   = K[0:X_s.shape[0], :]
    X_t_proj = K[X_s.shape[0]: , :]
    
    
    print("training...")
    madaotClf = Madaot(delta= 1, zeta= 1e-5)
    madaotClf.fit(X_s_proj, y_s, X_t_proj)
    
    X_test_proj = X_test.dot(projSpace.T).toarray()
    
    # sourceAccuracy["%s->%s"%(source,target)] = daFun.marginViolationLoss(X_s.dot(a), y_s, 0)
    # targetAccuracy["%s->%s"%(source,target)] = daFun.marginViolationLoss(X_t.dot(a), y_t, 0)
    testAccuracy["%s->%s"%(source,target)]   = accuracy_score(y_test, madaotClf.predict(X_test_proj))
    # print("\naccuracy on source: %.2f"%(100*(1-sourceAccuracy["%s->%s"%(source,target)])))
    # print("accuracy on target: %.2f"%(100*(1-targetAccuracy["%s->%s"%(source,target)])))
    print("accuracy on test: %.2f"%(100*testAccuracy["%s->%s"%(source,target)]))