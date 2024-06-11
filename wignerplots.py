# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:37:51 2024

@author: Gonzalez
"""

import qutip as qt
import matplotlib.pyplot as plt
from model import build_system_ident
import numpy as np
import matplotlib as mpl
from math import floor
from scipy.stats import linregress
from qutip.tensor import tensor
from time import time

def wigner1(rho,dim):
    
    """
    This function returns a plot of the wigner function for a given density matrix
    
    rho: qObj of the density matrix
    dim: the number of dimensions
    """
    rho1=rho.ptrace(0)
    
    rho2=rho.ptrace(1)
    
    
    xvec=np.linspace(-10, 10, 200)
    yvec=xvec
    fig,[ax1,ax2] = plt.subplots(1,2)
    for rho, ax in zip([rho1,rho2],[ax1,ax2]):  
        psi1 = qt.coherent(dim,5+5j)
        rho1 = psi1*psi1.dag()
        ax.set_aspect("equal")
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        W=qt.wigner(rho,xvec,yvec)
        #cmap = qt.matplotlib_utilities.wigner_cmap(W, max_color="Blue",
        #                                          min_color="Red",neg_color="#ff9698")
        
        cmap="magma"
        wmin, wmax = W.min(), W.max()
        #print(wmin,wmax)
        
        
        ax.pcolormesh(
            xvec, yvec, W, cmap=cmap, norm=mpl.colors.Normalize(wmin, wmax), shading="gouraud", rasterized=True
        )
    plt.show()
    
def steadywigner(H,Js,dim):
    
    """
    This function returns a plot of the wigner function of the steady state a given master equation.
    
    H: The Hamiltonian of the system
    Js: List of jump operators
    """
    time1=time()

    rho= qt.steadystate(H,Js,method="eigen",sparse=True,tol=3e-1,maxiter=200)

    rho1=rho.ptrace(0)
    rho2=rho.ptrace(1)
    time2=time()
    print(time2-time1)
    #rho2=rho.ptrace(1)

    xvec=np.linspace(-10, 10, 200)
    yvec=xvec
    
    #for rho in (rho1,rho2,rho):
    k=1 
    
    fig,[ax1,ax2] = plt.subplots(1,2)
    for rho, ax in zip([rho1,rho2],[ax1,ax2]):  
        ax.set_aspect("equal")
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        
        W=qt.wigner(rho,xvec,yvec)     
        
        cmap="magma"
        wmin, wmax = W.min(), W.max()
        #print(wmin,wmax)
        
        
        ax.pcolormesh(
            xvec, yvec, W, cmap=cmap, norm=mpl.colors.Normalize(wmin, wmax), shading="gouraud", rasterized=True
        )
        
        #with open(f"wignerplots\wigner{k}_{dim}","w") as f:
        #    f.write(str(W))
            
        k=2
    plt.show() 
    
    
def roundto1(x):
    return round(x,-int(floor(np.log10(abs(x)))))



        
if __name__ == "__main__":
    
    etalist=np.linspace(2.5,4.5,20)
    etay=[]
    """
    Vlist=np.linspace(1,2,10)    
    for V in Vlist:
        eta=equivwigner(0.1,0.5,etalist,0.1,V=V)
        etay.append(eta)
        print(eta)
        L=build_system_ident(0.1, 0.5, eta, 0.1,V=V,dim=dim, couple=1)
        
        steadywigner(L)
        plt.title(f"eta = {eta}")
    plt.scatter(Vlist,etay)
    REG= linregress(Vlist,etay)
    print(REG[0],REG[1])
    plt.title("eta vs V")
    """
    
    
    
    
    g1=0.1
    g2=0.2
    eta=0.7
    v=1
    couple=1

    dim = 15

    t0 = time()
    H, Js = build_system_ident(
        g1, g2, eta, 0.5, V=v, dim=dim, couple=couple, full_lv=False)
    
    
    
    steadywigner(H, Js, dim)
    print(time()-t0)
    