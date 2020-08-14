#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:39:40 2020

@author: dhouib
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
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
nPts= 400
nRep = 1
kernel= True

thetaDeg = 50
rotTheta=  np.deg2rad(thetaDeg)
rotMat = rotationMat(rotTheta)

X_s, y_s = datasets.make_moons(nPts, noise= 0.1)
y_s = 2*y_s-1
X_s = X_s - np.mean(X_s, axis = 0)

# Generating target related quantities with some preprocessing
shuffledIdx = shuffle(np.arange(len(X_s)))
X_t = np.dot(X_s[shuffledIdx], rotMat.T)
y_t = y_s[shuffledIdx]

gammaK = 0.5/np.mean(pdist(X_s, metric= "sqeuclidean")) # as suggested in the paper

projSpace = np.vstack((X_s, X_t))
K = rbf_kernel(projSpace, gamma= gammaK)
X_s_proj   = K[:nPts,:]
X_t_proj = K[nPts:, :]
    

ims = []
for delta in np.linspace(0, 1, 101):
    print("\nangle = %.0f"%thetaDeg)
    with open(resultsPath +"/theta%.0f/cross_valid"%thetaDeg, "rb") as resultsFile:
            results_raw = pkl.load(resultsFile)

    testScores = np.empty(nRep)
    sourceScores = np.empty(nRep)
    targetScores = np.empty(nRep)
    
    testOTPropagateScores = np.empty(nRep)
    sourceOTPropagateScores = np.empty(nRep)
    targetOTPropagateScores = np.empty(nRep)
    
    
    
    for i in range(nRep):
       

        
        
        bestArgsDict = da.postprocessing(results_raw, 1)
        bestArgsDict["zeta"] = 1e-5
        bestArgsDict["delta"]= delta
        print(bestArgsDict)
        
        
        madaot = Madaot(**bestArgsDict, nIter= 10)
        madaot.fit(X_s_proj, y_s, X_t_proj)
        
        testScores[i]   = accuracy_score(y_t, madaot.predict(X_t_proj))

    print(np.mean(testScores))
    
    tick_params_x = {"axis": "x", "which": "both", "top": False, "bottom": False, 
                   "labeltop": False, "labelbottom": False}
    tick_params_y = {"axis": "y", "which": "both", "left": False, "right": False, 
                   "labelleft": False, "labelright": False}
    pltNpts = 101
    gridRange = np.linspace(-2,2,pltNpts)
    adaptSurf = decisionFunc(gridRange, madaot.coefs, projSpace= projSpace)
    fig = plt.figure(figsize= (10,5))
    ax = plt.subplot(121)
    plt.imshow(adaptSurf, extent=(-2,2,-2,2) , origin= 'lower', cmap= 'RdBu')
    inds_s = np.argsort(y_s)
    
    ax.scatter(*X_s[inds_s,].T, c= ["r"]*(nPts//2) + ["b"]*(nPts//2), edgecolors= "k", alpha= 0.5)
    
    inds_t = np.argsort(y_t)
    ax.scatter(*X_t[inds_t,:].T,  c= ["r"]*200 + ["b"]*200, marker= 'v', edgecolors= "k")
    
    ax.legend(["Source", "Target"])
    
    ax.contour(gridRange, gridRange, adaptSurf, [0], colors= 'c', linewidths= 2)

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.tick_params(**tick_params_x) 
    ax.tick_params(**tick_params_y) 
    ax.set_title("decision boundary")
    
    ax2 = plt.subplot(122)
    from seaborn import distplot     
    oneD_s = madaot.decisionFunc(X_s_proj)
    oneD_t = madaot.decisionFunc(X_t_proj)
    
    distplot(oneD_s, color= 'b')
    distplot(oneD_t, color= 'r')
    ax2.set_title("$h(x)$ distribution")
    ax2.legend(["source", "target"])
    
    ax2.set_xlim([-10,10])
    ax2.set_ylim([0,0.4])
    ax2.tick_params(**tick_params_x) 
    ax2.tick_params(**tick_params_y)
    
    fig.suptitle("alignment term weight: %.2f"%delta)
    plt.savefig("angle%d_delta%.2f_hist.png"%(thetaDeg,delta), format= "png", bbox_inches='tight')