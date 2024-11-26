import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import integrate
from numpy import random
import pickle
import time

def construct_symmetric_block_matrix(matrices):
    """
    Constructs a square symmetric matrix with the specified block structure,
    accommodating matrices of varying sizes, including 1x1 matrices.
    
    Parameters:
        matrices (list of numpy.ndarray): A list of matrices [H_0, H_1, ..., H_n].
    
    Returns:
        numpy.ndarray: A square symmetric matrix with the specified block structure.
    """
    n = len(matrices)
    
    # Initialize lists for cumulative row and column sizes
    row_sizes = []
    col_sizes = []
    
    # Compute block sizes
    for i in range(n):
        # For each matrix H_i, append its row and column sizes
        row_sizes.append(matrices[i].shape[0])
        col_sizes.append(matrices[i].shape[1])
    
    # The size of the square matrix is the sum of all block sizes
    total_row_size = sum(row_sizes) + matrices[-1].shape[1]
    total_col_size = sum(col_sizes) + matrices[0].shape[0]
    
    # Initialize the square matrix with zeros
    symmetric_matrix = np.zeros((total_row_size, total_col_size))
    
    # Compute cumulative indices for row and column positions
    row_indices = [0]
    col_indices = [0]
    
    for size in row_sizes:
        row_indices.append(row_indices[-1] + size)
    for size in col_sizes:
        col_indices.append(col_indices[-1] + size)
    
    # Place the matrices and their transposes
    for i in range(n):
        # Upper subdiagonal block H_i
        row_start = row_indices[i]
        row_end = row_indices[i] + matrices[i].shape[0]
        col_start = col_indices[i] + matrices[0].shape[0]  # Offset by H_0's column size
        col_end = col_start + matrices[i].shape[1]
        
        symmetric_matrix[row_start:row_end, col_start:col_end] = matrices[i]
        
        # Lower subdiagonal block H_i.T
        symmetric_matrix[col_start:col_end, row_start:row_end] = matrices[i].T
    
    return symmetric_matrix

def bethe_dos(q,s):
    if abs(s)<2*np.sqrt(q):
        #print("check", abs(s), 2*np.sqrt(q))
        return (q+1)/(2*np.pi)*np.sqrt(4*q-s**2)/((q+1)**2-s**2)
    else:
        #print("check2", abs(s), 2*np.sqrt(q))
        return 0

class effective_Caylee2type_HL:
    
    def __init__(self,M,V,T,mu,p=8,q=2):
        self.q=q
        self.p=p
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu
        self.shells_size, self.d2, self.d1=self.make_shells_size()
        self.hops_matrices=self.make_hops()
        
        self.Delta=[]
    
    def local_DoS(self, energy):
        eps=0.01
        BdG_H=self.effective_H(0)
        spectra, vectors = eigh(BdG_H)
        rho=0
        for i in range(self.M+1):
            rho+=np.imag(vectors[0,i]*np.conj(vectors[0,i])/(energy -spectra[i]-1j*eps))         
        return rho/np.pi
    
    def make_shells_size(self):
        shells_size=[1,3,2*3,4*3,8*3-3,36]
        d2=[1,3,2*3,4*3,18,33]
        d1=shells_size-d2
        
        if self.M>5:
            
            while len(shells_size)<self.M+2:
                d2.append(2*d2[-1]-2*d2[-4]+d2[-5])
                shells_size.append(d2[-1]+d2[-5])
                d1.append(shells_size[-1]-d2[-1])
        
        #test to reproduce usual Caylee tree
        # shells_size=[1,3]
        # while len(shells_size)<self.M+2:
        #     shells_size.append(shells_size[-1]*2)
        
        return shells_size
    
    def make_hops(self):
        hops_array=[]
        hops_array.append([[np.sqrt(self.q)]])
        
        return hops_array
    
    #Fermi function
    def F(self,E):
        return 1/(np.exp((E)/self.T)+1)    
    
    def effective_H(self,k):
        
        #if k==0:
            
        #if k==1:
        
        if k>1:
            hops=np.zeros(self.M-k)
            for i in range(self.M-k):
                hops[i]=np.sqrt(self.shells_size[k+i+1]/self.shells_size[k+i])
            #print(hops)

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
                    norm=1/self.shells_size[i]
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k>0:
                for i in range(N):
                    norm=(self.shells_size[k]/self.shells_size[k-1]-1)*self.shells_size[k-1]/self.shells_size[i+k]
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

        plt.title("effective 1 type Caylee tree, q="+str(self.q)+", M="+str(self.M), fontsize=20)
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

class effective_Caylee_HL:
    
    def __init__(self,M,V,T,mu,p=8,q=2):
        self.q=q
        self.p=p
        self.M=M
        self.V=V
        self.T=T
        self.mu=mu
        self.shells_size=self.make_shells_size()
        self.Delta=[]
    
    def local_DoS(self, energy):
        eps=0.01
        BdG_H=self.effective_H(0)
        spectra, vectors = eigh(BdG_H)
        rho=0
        for i in range(self.M+1):
            rho+=np.imag(vectors[0,i]*np.conj(vectors[0,i])/(energy -spectra[i]-1j*eps))         
        return rho/np.pi
    
    def make_shells_size(self):
        shells_size=[1,3,2*3,4*3,8*3-3,36]
        d=[3,2*3,4*3,18,33]
        
        if self.M>5:
            
            while len(shells_size)<self.M+2:
                d.append(2*d[-1]-2*d[-4]+d[-5])
                shells_size.append(d[-1]+d[-5])
        
        # ##test to reproduce usual Caylee tree
        # shells_size=[1,3]
        # while len(shells_size)<self.M+2:
        #     shells_size.append(shells_size[-1]*2)
        
        return shells_size
    
    #Fermi function
    def F(self,E):
        return 1/(np.exp((E)/self.T)+1)    
    
    def effective_H(self,k):
        
        hops=np.zeros(self.M-k)
        for i in range(self.M-k):
            hops[i]=np.sqrt(self.shells_size[k+i+1]/self.shells_size[k+i])
        #print(hops)

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
                    norm=1/self.shells_size[i]
                    vectors_up[i,:]=self.V*norm*vectors_up[i,:]
            
            if k>0:
                for i in range(N):
                    norm=(self.shells_size[k]/self.shells_size[k-1]-1)*self.shells_size[k-1]/self.shells_size[i+k]
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

        plt.title("effective 1 type Caylee tree, q="+str(self.q)+", M="+str(self.M), fontsize=20)
        #plt.title("Bethe lattice DoS", fontsize=20)

        plt.show()
        plt.close()
    
    def plot_local_DoS(self):
        
        energies=np.linspace(-3, 3,100)
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
        
        energies=np.linspace(-3, 3,100)
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
    M=1000
    T=0.01
    V=1
    mu=0
    
    #CT=Caylee_tree(q, M, V, T, mu)
    # # CT.BdG_cycle()
    # # CT.plot_Delta()
   
    #CT.plot_local_DoS()
    
    #HL=effective_Caylee_HL(M, V, T, mu)
    # # HL.BdG_cycle()
    # # HL.plot_Delta()
   
    #HL.plot_local_DoS()
    
    #HL=effective_Caylee2type_HL(M, V, T, mu)
    # HL.BdG_cycle()
    # HL.plot_Delta()
   
    #HL.plot_local_DoS()
    
    H_0 = np.array([[1]])           # Shape (1, 2)
    H_1 = np.array([[2]])              # Shape (2, 1)     # Shape (1, 3)
    H_2 = np.array([[8,7]])
    H_3 = np.array([[3,4],[5,6]])  
    H_4 = np.array([[3,4],[5,6]])  


    
    matrices = [H_0, H_1, H_2,H_3,H_4]
    result = construct_symmetric_block_matrix(matrices)
    print(result)

    
main()