import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import integrate
from numpy import random
import pickle
import time

def bethe_dos(q,s):
    if abs(s)<2*np.sqrt(q):
        #print("check", abs(s), 2*np.sqrt(q))
        return (q+1)/(2*np.pi)*np.sqrt(4*q-s**2)/((q+1)**2-s**2)
    else:
        #print("check2", abs(s), 2*np.sqrt(q))
        return 0

class effective_Caylee_HL:
    
    def __init__(self,M,V,T,mu,p=8,q=3):
        self.q=q
        self.p=p
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu
        self.Delta=[]

    def kinetic_spectrum(self):
        spectrum=[]
        for k in range(self.M+1):
            H=self.effective_H(k)
            #print(np.shape(H), k)
            spectra, vectors = eigh(H)
            #print(np.round(spectra,4))
            if k==0:
                spectrum.append(spectra)
            if k==1:
                deg=self.q
                for j in range(deg):
                    spectrum.append(spectra)
            if k>1:
                deg=(self.q+1)*(self.q-1)*self.q**(k-2)
                for j in range(deg):
                    spectrum.append(spectra)
        spectrum=np.concatenate(spectrum)
        return np.sort(spectrum.flatten())
        
    def local_DoS(self, index):
        eps=0.05
        
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
        #print("k", k)
        #print("H", H)
       # print("Delta", Delta)
        BdG_H = np.block([[H, Delta], [Delta, -H]])
        #print(BdG_H)
        return BdG_H
    
    def gap_integral(self,Delta):
        
        gap=np.zeros(self.M+1)
        
        for k in range(self.M+1):
            N=self.M+1-k
            #print("N",N)
            BdG_H=self.effective_BdG(k,Delta[k:])
            spectra, vectors = eigh(BdG_H)
            F_weight=np.ones(N)-2*self.F(spectra[N:])
            vectors_up=np.copy(vectors[N:,N:])
            vectors_down=np.copy(vectors[:N,N:])
                    
            if k==0:
                for i in range(N):
                    if i==0:
                        norm=1
                    if i>0:
                        norm=1/(self.q+1)*self.q**(-(i-1))
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k==1:
                for i in range(N):
                    norm=1/(self.q+1)*self.q**(-(i-1))
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k>1:
                for i in range(N):
                    norm=(self.q-1)*self.q**(-i-1)
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
                    
            gap[k:]+=np.einsum(vectors_up, [0,1], vectors_down, [0,1], F_weight,[1],[0])

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
        
        self.Delta=Delta
        return Delta
    
    def plot_Delta(self):
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel(r'$\Delta$',fontsize=20)
        plt.xlabel(r'distance from the center',fontsize=20)
        plt.plot(self.Delta)

        plt.title("Caylee tree, q="+str(self.q)+", M="+str(self.M), fontsize=20)
        #plt.title("Bethe lattice DoS", fontsize=20)

        plt.show()
        plt.close()

class Caylee_tree:
    
    def __init__(self,q,M,V,T,mu):
        self.q=q
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu
        self.Delta=[]
    
    def kinetic_spectrum(self):
        spectrum=[]
        for k in range(self.M+1):
            H=self.effective_H(k)
            #print(np.shape(H), k)
            spectra, vectors = eigh(H)
            #print(np.round(spectra,4))
            if k==0:
                spectrum.append(spectra)
            if k==1:
                deg=self.q
                for j in range(deg):
                    spectrum.append(spectra)
            if k>1:
                deg=(self.q+1)*(self.q-1)*self.q**(k-2)
                for j in range(deg):
                    spectrum.append(spectra)
        spectrum=np.concatenate(spectrum)
        return np.sort(spectrum.flatten())
        
    def local_DoS(self, energy):
        eps=0.01
        BdG_H=self.effective_H(0)
        spectra, vectors = eigh(BdG_H)
        rho=0
        for i in range(self.M+1):
            rho+=np.imag(vectors[0,i]*np.conj(vectors[0,i])/(energy -spectra[i]-1j*eps))         
        return rho/np.pi
    
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
        #print("k", k)
        #print("H", H)
       # print("Delta", Delta)
        BdG_H = np.block([[H, Delta], [Delta, -H]])
        #print(BdG_H)
        return BdG_H
    
    def gap_integral(self,Delta):
        
        gap=np.zeros(self.M+1)
        
        for k in range(self.M+1):
            N=self.M+1-k
            #print("N",N)
            BdG_H=self.effective_BdG(k,Delta[k:])
            spectra, vectors = eigh(BdG_H)
            F_weight=np.ones(N)-2*self.F(spectra[N:])
            vectors_up=np.copy(vectors[N:,N:])
            vectors_down=np.copy(vectors[:N,N:])
                    
            if k==0:
                for i in range(N):
                    if i==0:
                        norm=1
                    if i>0:
                        norm=1/(self.q+1)*self.q**(-(i-1))
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k==1:
                for i in range(N):
                    norm=1/(self.q+1)*self.q**(-(i-1))
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k>1:
                for i in range(N):
                    norm=(self.q-1)*self.q**(-i-1)
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
                    
            gap[k:]+=np.einsum(vectors_up, [0,1], vectors_down, [0,1], F_weight,[1],[0])

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
        
        self.Delta=Delta
        return Delta
    
    def plot_Delta(self):
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel(r'$\Delta$',fontsize=20)
        plt.xlabel(r'distance from the center',fontsize=20)
        plt.plot(self.Delta)

        plt.title("Caylee tree, q="+str(self.q)+", M="+str(self.M), fontsize=20)
        #plt.title("Bethe lattice DoS", fontsize=20)

        plt.show()
        plt.close()
    
    def plot_local_DoS(self):
        
        energies=np.linspace(-3.5, 3.5,100)
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel(r'$\rho(s)$',fontsize=20)
        plt.xlabel(r's',fontsize=20)
        plt.plot(energies, self.local_DoS(energies))
        
        limit_dos=[]
        for energy in energies:
            limit_dos.append(bethe_dos(self.q, energy))
        plt.plot(energies,limit_dos)

        plt.title("local DoS Caylee tree, q="+str(self.q)+", M="+str(self.M), fontsize=20)
        #plt.title("Bethe lattice DoS", fontsize=20)

        plt.show()
        plt.close()
        
    
def main():
    q=2
    M=50
    T=0.01
    V=1
    mu=0
    
    CT=Caylee_tree(q, M, V, T, mu)
    CT.BdG_cycle()
    CT.plot_Delta()
   
    # CT.plot_local_DoS()
    

    
main()