import numpy as np
import qutip as qt
from scipy.linalg import eig
from scipy.optimize import root
import scipy.sparse as sps
from scipy.linalg import block_diag
#from constants import DIM_N
from utils import local_data_path, build_datapath
import matplotlib.pyplot as plt
from qutip.tensor import tensor
from optimize import approxeigen
from time import time
from scipy.sparse.linalg import eigs
def build_system_ident(g1, g2, eta, D, V=1, n=2, m=2, dim=2, full_lv=True, phi=0, adag=False, interaction=True,couple=1):
    
    """
    This function returns the Liouvillian for a given set of parameters for thwo identical coupled dissipative oscillators.
    
    
    g1, g2: dissipation amplitudes
    
    eta: squeezing
    
    D difference between oscilaltor and squeezing frequencies. For the reactive model D is directly omega_s
    
    V: interaction force, V_delta or V_rho for each model
    
    n, m: squeezing and dissipation orders
    
    dim: number of dimensions
    
    full_lv: if True the function returns the full liouvillian, else, it returns H and a list of jump operators
    
    interaction: if the model takes into account any interaction model
    
    couple: if 1, we use the dissipative model, if 2 we use the reactive model
    """
    
    a0=qt.destroy(dim)
    
    Z=qt.identity(dim)
    a=tensor(a0,Z)
    
    b=tensor(Z,a0)
    phase=np.exp(1j*phi*n)
    if couple==1:
        H = D*a.dag()*a + D*b.dag()*b
    
    elif couple==2:
        H = V*(a.dag()*b+b.dag()*a) + D*(a.dag()*a-b.dag()*b)
    H +=  1j*eta*(a**n*phase - a.dag()**n*np.conj(phase))
    H +=  1j*eta*(b**n*phase - b.dag()**n*np.conj(phase))
    
    Js=[]
    
    if g1 > 0:
        Js.append(np.sqrt(g1) * (a.dag() if adag else a))
        Js.append(np.sqrt(g1) * (b.dag() if adag else b))
    if g2 > 0:
        Js.append(np.sqrt(g2) * a**m)
        Js.append(np.sqrt(g2) * b**m)
        
    if interaction and couple==1:
        Js.append(V*(a-b))
        
    return qt.liouvillian(H, Js) if full_lv else (H, Js)


#eivals,_,_=eigendecomposition(1,20,1,10,V=1,dim=25,n_eigv=10,raw=True)
#1,20,1,10,V=10 ,dim=25
def plots(couple,marker,N,V,eta,g2,dim,plot=True,g1=0.1):
    
    """
    This function returns the value of k and a plot of the eigenvalues in the complex plane, calculated for the given parameters. k = t5/t4
    
    
    couple: if 1, we use the dissipative model, if 2 we use the reactive model
    
    marker: the type of marjer used to plot
    
    N: the number of eigenvalues to calculate. can be an int or a list of int
   
    V: the interaction strength. Can be a float or a list of floats
    
    eta, g1, g2: squeezing and dissipation amplitudes
    
    plot: if true, the plot is shown. Else, only the value of k is returned
    """
    
    Data=[]
    if isinstance(V, float) or isinstance(V,int):
        V=[V]
    if isinstance(N,int):
        N=[N]
        
    for v in V:
        L=build_system_ident(g1, g2, eta, 0.5, V=v, dim=dim, couple=couple)
        try:
            eivals=L.eigenenergies(sort="high",eigvals=max(N),sparse=True,maxiter=2000,tol=1e-2 )
        except:
            eivals=[1,1,1,1,1]
        R=[]
        I=[]
        for i in eivals: 
            R.append(np.real(i))
            I.append(np.imag(i))
            
        Data.append([R,I])
    if plot:
        fig,[ax1,ax2]=plt.subplots(2)    
        for n in N:
            Leg=[]
            for i,Param in enumerate(Data):
                if n>=10:
                    ax1.scatter(Param[0][:n],Param[1][:n],marker=marker[i])
                else:
    
                    ax2.scatter(Param[0][:n],Param[1][:n],marker=marker[i])
                Leg.append(r"$V_\rho$ ="+f" {V[i]}; "+r"$\kappa$"+f" = {round(Param[0][4]/Param[0][3])}")
            
            ax2.set_ylim(-0.1,0.1)
            """
            l=2*min(min(min(Data)))
            ax2.set_xlim(left=-0.0033,right=-0.1*l)
            """
            ax2.set_xlim(-0.48,0.03)
        if couple==1:
            ax1.set_title(f"Eigenvalues of the dissipative model")
        elif couple==2:
            ax1.set_title("Eigenvalues of the reactive model")
             
        ax2.legend(Leg)
        
        ax1.set_ylabel(r"$Im\{\lambda\}$")
        ax2.set_ylabel(r"$Im\{\lambda\}$")
        ax2.set_xlabel(r"$Re\{\lambda\}$")
        ax2.set_yticks([-0.1,0,0.1])
        plt.show()
    else:
        Param=Data[-1] 
        
    k=Param[0][4]/Param[0][3]
    #k2 = Param[0][2]/Param[0][1]
    return k
if __name__ == "__main__":
    
    #To make the k vs n plots
   
    dimlist = [16,18,20,22,24,26,28,30]
    klist = []
    timeslist=[]
    
    etas = [0.7]
    ges = [0.2]
    K=[]
    for g2, eta in zip(ges,etas):
        
        klist = []
        timeslist=[]
        
        for i in dimlist:
                print(i)
                t0=time()
                klist.append(plots(1,["o"],[5],V=0.5,dim=i,g2=g2,eta=eta, plot=False))
                dt=time()-t0
                timeslist.append(dt)
                print(f"time elapsed:{dt}")
        print(klist)
        print(timeslist)
        plt.scatter(dimlist,klist)
        plt.show()
        K.append(klist)
    
    
    #To make the k vs V plots
    """
    
    vlist=np.logspace(-5,-3,5)
    dim=30
    klist=[]
    for v in vlist:
        t0=time()
        klist.append(plots(2,["o"],[5],V=v,dim=dim,g2=0.2,eta=0.7,plot=False))
        dt=time()-t0

        print(f"time elapsed:{dt}")
        print(klist[-1])
    print(klist)

    plt.scatter(vlist,klist)
    """
    
    # To make the 2D plot of g2 and eta
    """
    
    dim=29
    etalist=np.linspace(0.3,0.8,20)    
    g2list=np.linspace(0.05,0.3,15)
    
    etalist = etalist[12:]
    g2list = g2list[:3]
    VALUES=[]
    V2 = []
    t0=time()
    telltime=True
    itr=1
    for g2 in g2list:
        fila=[]
        f2 = []
        for eta in etalist:
            v,k2=plots(2,["o"],5,V=0.5    ,eta=eta,g2=g2,dim=dim,plot=False)
            if telltime:
                tf=time()
                print(f"Tiempo transurrido: {tf-t0} s")
            fila.append(v)
            f2.append(k2)
            print(f"eta = {eta}, g2 = {g2}, k = {v}")
            print(f"iteracion {itr}/{len(g2list)*len(etalist)}")
            itr +=1
            print()
        
        VALUES.append(fila)
        V2.append(f2)
    plt.xlabel("eta")
    plt.ylabel("g2")
    print(VALUES)
    with open("k_V=0.02.txt","w") as f:
        f.write(str(VALUES))
    plt.pcolormesh(etalist,g2list,VALUES,cmap="magma")
    #plt.gca().set_aspect('equal')

    
    dim = 20
    plots(1,["o","+","x"],[12,4],V=(0.1,0.5,0.75),eta=0.6,g2=0.2,dim=dim)
    """

