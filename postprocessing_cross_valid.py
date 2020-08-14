#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sys import path as path
from os import path, makedirs
from scipy.spatial.distance import pdist
import myDA as da 
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import rbf_kernel
from madaot import Madaot
from sklearn.metrics import accuracy_score
resultsPath = "results/DA/"

if not path.exists(resultsPath):
    makedirs(resultsPath)
    
import pickle as pkl

plt.rcParams.update({"font.family":"serif",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{serif}",
         r"\usepackage{amsmath, amsfonts, amssymb, amstext, amsthm, bbm, mathtools}",
         ]
})

plt.rc('text', usetex=True)

plt.close('all')



#%%
def rotationMat(rotTheta):
    return np.array([[np.cos(rotTheta), -np.sin(rotTheta)], 
                     [np.sin(rotTheta), np.cos(rotTheta)]])
def decisionFunc(gridRange, v, projSpace, transformer= None):
    pltNpts = len(gridRange)
    X_grid = np.vstack((np.tile(gridRange, pltNpts), np.repeat(gridRange, pltNpts))).T
    
    if projSpace is None:
        X_grid_t = transformer.transform(X_grid)
    else:
        X_grid_t = rbf_kernel(X_grid, projSpace, gamma= gammaK)
    return np.dot(X_grid_t, v).reshape(pltNpts, pltNpts)

#%%
nPts= 300
nRep = 1
kernel= True

for thetaDeg in np.arange(10, 100, 10):
    print("\nangle = %.0f"%thetaDeg)
    with open(resultsPath +"/theta%.0f/cross_valid"%thetaDeg, "rb") as resultsFile:
            results_raw = pkl.load(resultsFile)

    testScores = np.empty(nRep)
    sourceScores = np.empty(nRep)
    targetScores = np.empty(nRep)
    
    testOTPropagateScores = np.empty(nRep)
    sourceOTPropagateScores = np.empty(nRep)
    targetOTPropagateScores = np.empty(nRep)
    
    rotTheta=  np.deg2rad(thetaDeg)
    rotMat = rotationMat(rotTheta)
    
    for i in range(nRep):
        X_s, y_s = datasets.make_moons(nPts, noise= 0.1)
        y_s = 2*y_s-1
        X_s = X_s - np.mean(X_s, axis = 0)
        
        # Generating target related quantities with some preprocessing
        shuffledIdx = shuffle(np.arange(len(X_s)))
        X_t = np.dot(X_s[shuffledIdx], rotMat.T)
        y_t = y_s[shuffledIdx]
        
        X_test, y_test = datasets.make_moons(1000, noise= 0.1)
        y_test = 2*y_test-1
        X_test = X_test - np.mean(X_test, axis = 0)
        X_test = np.dot(X_test, rotMat.T)

        gammaK = 0.5/np.mean(pdist(X_s, metric= "sqeuclidean")) # as suggested in the paper
        
        bestArgsDict = da.postprocessing(results_raw, 1)
        bestArgsDict["delta"] = 1
        bestArgsDict["zeta"] = 1e-5
        
        print(bestArgsDict)
        projSpace = np.vstack((X_s, X_t))
        K = rbf_kernel(projSpace, gamma= gammaK)
        X_s_proj   = K[:nPts,:]
        X_t_proj = K[nPts:, :]
        X_test_proj = rbf_kernel(X_test, projSpace, gamma= gammaK)
        
        madaot = Madaot(**bestArgsDict)
        madaot.fit(X_s_proj, y_s, X_t_proj)
        
        testScores[i]   = accuracy_score(y_test, madaot.predict(X_test_proj))

    print(np.mean(testScores))
    
    
    pltNpts = 101
    gridRange = np.linspace(-2,2,pltNpts)
    adaptSurf = decisionFunc(gridRange, madaot.coefs, projSpace= projSpace)
    fig = plt.figure()
    plt.imshow(adaptSurf, extent=(-2,2,-2,2) , origin= 'lower', cmap= 'RdBu')
    inds_s = np.argsort(y_s)
    plt.scatter(*X_s[inds_s,].T, c= ["r"]*(nPts//2) + ["b"]*(nPts//2), edgecolors= "k", alpha= 0.5)
    
    
    inds_t = np.argsort(y_t)
    plt.scatter(*X_t[inds_t,].T, c= ["r"]*(nPts//2) + ["b"]*(nPts//2), edgecolors= "k", marker= "v")
    # inds_test = np.argsort(y_test)
    # plt.scatter(*X_test[inds_test,:].T,  c= ["r"]*500 + ["b"]*500, marker= 'v', edgecolors= "k")
    
    plt.legend(["Source", "Target"])
    
    contour2 = plt.contour(gridRange, gridRange, adaptSurf, [0], colors= 'c', linewidths= 2)

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    tick_params_x = {"axis": "x", "which": "both", "top": False, "bottom": False, 
                   "labeltop": False, "labelbottom": False}
    tick_params_y = {"axis": "y", "which": "both", "left": False, "right": False, 
                   "labelleft": False, "labelright": False}
    plt.tick_params(**tick_params_x) 
    plt.tick_params(**tick_params_y) 
    # plt.savefig("angle%d.pgf"%thetaDeg, format= "pgf", bbox_inches='tight')
    plt.savefig("angle%d.pdf"%thetaDeg, format= "pdf", bbox_inches='tight')