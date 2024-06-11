# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:20:09 2024

@author: Gonzalez
"""

import qutip as qt
import numpy as np
from model import build_system_ident
from qutip.tensor import tensor
import matplotlib.pyplot as plt


plt.rcParams["figure.autolayout"] = True

fig, [ax1, ax2] = plt.subplots(1,2)

g1=0.1
g2=0.2
eta=0.3
v=0.5
couple=1
dim=27
Ais = []
Bis = []

for v in [0.1]:
    H,Js=build_system_ident(g1, g2, eta, 0.5, V=v,
                            dim=dim, couple=couple,
                            full_lv=False)
    
    a0=qt.destroy(dim)
    Z=qt.identity(dim)
    
    a=tensor(a0,Z)
    b=tensor(Z,a0)
    
    r1 = 0.7
    r2 = 0.7
    th1=np.pi/2-0.2
    
    th2 = -np.pi/2+0.4
    
    
    psi01 = qt.coherent(dim,  r1*np.exp(1j*th1))
    psi02 = qt.coherent(dim, r2*np.exp(1j*th2))
    
    psi0 = tensor(psi01,psi02)
    
    
    N = 1000
    tlist = np.linspace(0,1000,N)
    tlist = np.logspace(-2,2.2,N)
    options = qt.Options(order=3,tidy = True,rtol=1e-2,atol=1e-2)
    [A,B] = qt.mesolve(H,psi0,tlist, Js, [a,b],options=options,progress_bar=True).expect    
    
    Ar = np.real(A)
    Ai = np.imag(A)
    
    Br = np.real(B)
    Bi = np.imag(B)
    
    Am = np.absolute(A)
    Bm = np.absolute(B)
    

    
    
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz
    
    for i in range(1):
        winsound.Beep(freq, duration)
        
    
    
    ax1.scatter(Ar,Ai, s = 4)
    ax1.scatter(Br,Bi, s = 4)
    ax1.set_aspect(1)
    ax1.set_xlabel(r"$Re\{\langle\hat{a_i}\rangle\}$")
    ax1.set_ylabel(r"$Im\{\langle\hat{a_i}\rangle\}$")
    ax1.set_yticks([-2,-1,0,1,2])
    ax1.grid(c=(0.8,0.8,0.8))
    lim=3
    ax1.set_xlim(-lim,lim)
    ax1.set_ylim(-lim,lim)  
    
    
    tlist=tlist*0.1
    
    ax2.plot(tlist, Ai)
    ax2.plot(tlist, Bi)
    ax2.set_xscale("log")
    ax2.set_ylabel(r"$|\langle\hat{a_i}\rangle|$")
    ax2.set_xlabel(r"$\gamma_1 t$")
    
    plt.show()
    
    Ais.append(Ai)
    Bis.append(Bi)

