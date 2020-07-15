#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import datasets
from sklearn.metrics.pairwise import rbf_kernel
from os import  makedirs, path
from myDA import DaGridSearchCV
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from time import sleep
from scipy.spatial.distance import pdist
from madaot import Madaot

resultsPath = "results/DA/"

if not path.exists(resultsPath):
    makedirs(resultsPath)
    
import pickle as pkl
#%%
nPts= 300
def rotationMat(rotTheta):
    return np.array([[np.cos(rotTheta), -np.sin(rotTheta)], 
                     [np.sin(rotTheta), np.cos(rotTheta)]])
#%% 
crossValidator = KFold(n_splits = 5)
nRep = 1
params = {
            "delta": [1],#np.logspace(-2,2,10),
            "zeta": np.logspace(-2,-6,10),
            }
for thetaDeg in np.arange(10,100,10): # for different angles
    sleep(1)
    print("\n\nAngle = %.f"%thetaDeg)
    rotTheta=  np.deg2rad(thetaDeg)
    rotMat = rotationMat(rotTheta)
    currentResPath = resultsPath + "/theta%.0f"%thetaDeg
    if not path.exists(currentResPath):
        makedirs(currentResPath)
    
    results_raw = [] # a list of lists, each list contains scores for the current repetition
    for i in range(nRep):
        sleep(0.1)
        print("iteration = %d"%i)
       # Generating source related quantities with some preprocessing
        X_s, y_s = datasets.make_moons(nPts, noise= 0.1)
        X_s = X_s - np.mean(X_s, axis = 0)
        

        y_s= 2*y_s-1
        
        # Generating target related quantities with some preprocessing
        shuffledIdx = shuffle(np.arange(len(X_s)))
        X_t = np.dot(X_s[shuffledIdx], rotMat.T)
        y_t = y_s[shuffledIdx]

        d = 2
        gammaK = 0.5/np.mean(pdist(X_s, metric= "sqeuclidean")) # as suggested in the paper
        K = rbf_kernel(np.vstack((X_s, X_t)), gamma= gammaK)
        
        results_raw = DaGridSearchCV(Madaot, X_s, y_s, X_t, y_t, param_grid= params, cv= crossValidator, 
                                     n_jobs= -1, verbose= 10, transformer= None, K= K, 
                                     reverse_cv= False)
    
    toSave = {"params": params, "nSplits": crossValidator.get_n_splits(), "scores_raw": results_raw}    
    with open(currentResPath + "/cross_valid", "wb") as file:
        pkl.dump(toSave, file, protocol= pkl.HIGHEST_PROTOCOL)