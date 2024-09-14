import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import integrate
from numpy import random
import pickle
import time

""

class effective_HL:
    
    def __init__(self,M,V,T,mu,p=8,q=3):
        self.q=q
        self.p=p
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu

class Caylee_tree:
    
    def __init__(self,q,M,V,T,mu):
        self.q=q
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu
    
    def kinetic_spectrum(self):
        spectrum=[]
        for k in range(self.M+1):

            H=self.effective_H(k)
            spectra, vectors = eigh(H)
            spectrum.append(spectra)
        spectrum=np.concatenate(spectrum)
        return np.sort(spectrum.flatten())
        
    def local_DoS(self):
        return
    
    #Fermi function
    def F(self,E):
        return 1/(np.exp((E)/self.T)+1)    
    
    def effective_H(self,k):
        
        hops=np.sqrt(self.q)*np.ones(self.M-k)
        if k==0:
            hops[0]=np.sqrt(self.q+1)
            H=np.diag(hops,k=1)+np.diag(hops,k=-1)
        else:
            H=np.diag(hops,k=1)+np.diag(hops,k=-1)
        return H
    
    def effective_BdG(self, k,Delta_k):
        H=self.effective_H(k)-self.mu*np.eye(self.M-k+1)
        Delta=np.diag(Delta_k)
        print("k", k)
        print("H", H)
        print("Delta", Delta)
        BdG_H = np.block([[H, Delta], [Delta, -H]])
        print(BdG_H)
        return BdG_H
    
    def gap_integral(self,Delta):
        
        gap=np.zeros(self.M+1)
        
        for k in range(self.M+1):
            N=self.M-k+1
            print("N",N)
            BdG_H=self.effective_BdG(k,Delta[k:])
            spectra, vectors = eigh(BdG_H)
            F_weight=np.ones(N)-2*self.F(spectra[N:])
            if k==0:
                vectors_up=self.V * np.conj(vectors[N:,N:])
            else:
                vectors_up=self.V * np.conj(vectors[N:,N:])
            gap[k:]+=np.einsum(vectors_up, [0,1], vectors[:N,N:], [0,1], F_weight,[1],[0])
        # k=0
        # N=self.M-k+1
        # print("N",N)
        # BdG_H=self.effective_BdG(k,Delta[k:])
        # spectra, vectors = eigh(BdG_H)
        # F_weight=np.ones(N)-2*self.F(spectra[N:])
        # vectors_up=self.V * np.conj(vectors[N:,N:])

        # gap[k:]+=np.einsum(vectors_up, [0,1], vectors[:N,N:], [0,1], F_weight,[1],[0])
            
        return gap
    
        
    def BdG_cycle(self):
        
        step=0
        Delta=0.5*np.ones(self.M+1)+0.1*np.random.rand(self.M+1)             
        while True:
            Delta_next=self.gap_integral(Delta)
            error=np.max(np.abs((Delta-Delta_next)))
            Delta=Delta_next
            print("step", step, "error", error, "Delta_max", np.max(np.abs(Delta)))
            step += 1
            if error<10**(-6):
                break
        
        return Delta
    
    
def main():
    q=2
    M=4
    T=0.1
    V=1
    mu=0
    CT=Caylee_tree(q, M, V, T, mu)
    spectrum=CT.kinetic_spectrum()
    print(np.round(spectrum,4))
    # Delta=CT.BdG_cycle()
        
    # fig, ax = plt.subplots(figsize=(9.6,7.2))
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.ylabel(r'$\Delta$',fontsize=20)
    # plt.xlabel(r'distance from the center',fontsize=20)
    # plt.plot(Delta)

    # plt.title("Caylee tree, q="+str(q)+", M="+str(M), fontsize=20)
    # #plt.title("Bethe lattice DoS", fontsize=20)

    # plt.show()
    # plt.close()
    
main()