#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numba import vectorize, float64
import numpy as np
from matplotlib import pyplot as plt

@vectorize([float64(float64, float64, float64)], nopython= True)
def loss(x, rho, beta):
    if beta != 0:
        if x < rho: return 1
        elif x< beta + rho: return 1-(x-rho)/beta
        else: return 0
    else:
        if x < rho: return 1
        else: return 0
        
rho  = 0.23
beta = 0.35
vec = np.linspace(-2,2,1001)

fig = plt.figure()

plt.plot(vec, loss(vec, rho, 0), linestyle= "dashed", linewidth= 3, label= r"$l^{\rho,0}$", alpha = 0.8)
plt.plot(vec, loss(vec, rho + beta, 0), linestyle= "dashdot", linewidth= 3, label= r"$l^{\rho+\beta,0}$", alpha = 0.8)
plt.plot(vec, loss(vec, rho, beta), linestyle= "solid", linewidth= 3, label= r"$l^{\rho,\beta}$", alpha = 0.8)

plt.scatter(rho, 0, c='k', marker= 'o', linewidth= 2)
plt.scatter(rho+beta, 0, c= 'k', marker= 'o', alpha= 1, linewidth= 2)
plt.text(rho, -0.1, r"$\rho$")
plt.text(rho + beta-0.1, -0.1, r"$\rho + \beta$")

plt.grid('minor')
plt.ylim([-0.5, 1.5])
plt.xlim([-1.5, 1.5])

plt.legend()
plt.axhline(y=0, xmin=-1, xmax=1, color= 'k', linewidth= 1)
plt.axvline(x=0, ymin=-0.5, ymax=1.5, color= 'k', linewidth= 1)
plt.axes().set_aspect('equal')
plt.savefig("loss_func.pdf", format= "pdf", bbox_inches='tight')
plt.tight_layout()