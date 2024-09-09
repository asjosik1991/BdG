import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import integrate
from numpy import random
import pickle
import time

""
#Fermi function
def F(E,T):
    return 1/(np.exp((E)/T)+1)    

def effective_H(k,q,M):
    
    hops=np.sqrt(q)*np.ones(M-k)
    if k==0:
        hops[0]=np.sqrt(q+1)
        H=np.diag(hops,k=1)+np.diag(hops,k=-1)
    else:
        H=np.diag(hops,k=1)+np.diag(hops,k=-1)
    return H

def effective_BdG(k,q,M, Delta_k,mu):
    H=effective_H(k,q,M)-mu*np.eye(M-k+1)
    Delta=np.diag(Delta_k)
    BdG_H = np.block([[H, Delta], [Delta, -H]])
    return BdG_H

def gap_integral(q,M,mu,T,V,Delta):
    
    gap=np.zeros(M+1)
    
    for k in range(M+1):
        N=M-k+1
        BdG_H=effective_BdG(k,q,M, Delta[k:],mu)
        spectra, vectors = eigh(BdG_H)
        F_weight=np.ones(N)-2*F(spectra[N:],T)
        if k==0:
            vectors_up=V * np.conj(vectors[N:,N:])
        else:
            vectors_up=q*V * np.conj(vectors[N:,N:])
        gap[k:]+=np.einsum(vectors_up, [0,1], vectors[:N,N:], [0,1], F_weight,[1],[0])
    return gap

    
def BdG_cycle(q,M,mu,T,V):
    
    step=0
    Delta=0.5*np.ones(M+1)+0.1*np.random.rand(M+1)             
    while True:
        Delta_next=gap_integral(q,M,mu,T,V,Delta)
        error=np.max(np.abs((Delta-Delta_next)))
        Delta=Delta_next
        print("step", step, "error", error, "Delta_max", np.max(np.abs(Delta)))
        step += 1
        if error<10**(-6):
            break
    
    return Delta

    
def main():
    q=3
    M=50
    T=1
    V=0.1
    mu=0
    Delta=BdG_cycle(q,M,mu,T,V)
        
    fig, ax = plt.subplots(figsize=(9.6,7.2))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\Delta$',fontsize=20)
    plt.xlabel(r'distance from the center',fontsize=20)
    plt.plot(Delta)

    plt.title("Caylee tree, q="+str(q)+", M="+str(M), fontsize=20)
    #plt.title("Bethe lattice DoS", fontsize=20)

    plt.show()
    plt.close()
    
main()