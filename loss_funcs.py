#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numba import vectorize, float64
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         r"\usepackage{amsmath, amsfonts, amssymb, amstext, amsthm, bbm, mathtools}",
         ]
})

@vectorize([float64(float64, float64, float64)], nopython= True)
def loss(x, rho, beta):
    if beta != 0:
        if x < rho: return 1
        elif x< beta + rho: return 1-(x-rho)/beta
        else: return 0
    else:
        if x < rho: return 1
        else: return 0
        
# rho  = 0.23
# beta = 0.35
# vec = np.linspace(-2,2,1001)


# tick_params_x = {"axis": "x", "which": "both", "top": False, "bottom": False, 
#                    "labeltop": False, "labelbottom": False}
# tick_params_y = {"axis": "y", "which": "both", "left": False, "right": False, 
#                    "labelleft": False, "labelright": False}
    
# fig = plt.figure()

# plt.plot(vec, loss(vec, rho, 0), linestyle= "dashed", linewidth= 3, label= r"$\ell_{\rho,0}$", alpha = 0.8, color= [0.33]*3)
# plt.plot(vec, loss(vec, rho + beta, 0), linestyle= "dashdot", linewidth= 3, label= r"$\ell_{\rho+\beta,0}$", alpha = 0.8, color= [0.66]*3)
# plt.plot(vec, loss(vec, rho, beta), linestyle= "solid", linewidth= 3, label= r"$\ell_{\rho,\beta}$", alpha = 0.8, color= [0]*3)

# plt.scatter(rho, 0, c='k', marker= 'o', linewidth= 2)
# plt.scatter(rho+beta, 0, c= 'k', marker= 'o', alpha= 1, linewidth= 2)
# plt.text(rho, -0.1, r"$\rho$", fontsize= "x-large")
# plt.text(rho + beta-0.1, -0.1, r"$\rho + \beta$", fontsize= "x-large")

# # plt.grid('minor')
# plt.ylim([-0.5, 1.5])
# plt.xlim([-1.5, 1.5])

# plt.legend(fontsize= "x-large")
# plt.axhline(y=0, xmin=-1, xmax=1, color= 'k', linewidth= 1)
# plt.axvline(x=0, ymin=-0.5, ymax=1.5, color= 'k', linewidth= 1)
# plt.axes().set_aspect('equal')

# plt.tight_layout()

# plt.tick_params(**tick_params_x)
# plt.tick_params(**tick_params_y)
# plt.xlabel(r"$t$", fontsize= "x-large")
# plt.ylabel(r"$\ell(t)$", fontsize= "x-large")


# plt.savefig("loss_func.pdf", format= "pdf", bbox_inches='tight')
# plt.savefig("loss_func.pgf", format= "pgf", bbox_inches='tight')

