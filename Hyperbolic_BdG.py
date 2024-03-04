import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib import cm, rc, ticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.sparse import diags
from numpy import random
import pickle
import time
import matplotlib.colors as mcolors
import networkx as nx
import math

"Class for initial one-particle lattice and corresponding hamiltonian"
class HyperLattice:
    def __init__(self,p,q,l,hopping):
        self.p=p
        self.q=q
        self.l=l
        self.hopping=hopping
        self.sites=np.array([])
        self.hamiltonian=[]
        self.create_hyperbolic_lattice()
 
    
    
    def gamma(self, z: np.complex128) -> np.complex128:
        sigma = math.sqrt( (math.cos(2*math.pi / self.p) + math.cos(2*math.pi / self.q)) / (1 + math.cos(2*math.pi / self.q)) )
        z = np.complex128(z)
        result = (z + sigma) / (sigma * z + 1)
        return result
    
    def rot(self, z, n):
        z = np.complex128(z)
        result = (math.cos(2 * math.pi * n / self.p) + math.sin(2 * math.pi * n / self.p) * 1j) * z
        return result
        
    def trans(self, z, n):
        result = self.rot(z,-n)
        result = self.gamma(result)
        result = self.rot(result, n)
        return result
    
    # It is useful to define the distance function    
    def dist(self, x, y):
        return math.acosh(1 + 2*abs(x-y)**2 / (1 - abs(x)**2) / (1-abs(y)**2))
   
        
    def create_hyperbolic_lattice(self):
    
        r0 = math.sqrt( math.cos(math.pi*(1/self.p +1/self.q) ) / math.cos(math.pi*(1/self.p - 1/self.q) ) )
    
        #Unit cell points
    
        for n in range(self.p):
            self.sites = np.append(self.sites, r0*math.cos(math.pi*(1+2*n)/self.p) + r0*math.sin(math.pi*(1+2*n)/self.p)*1j )
    
    
        # Next, we generate new cells by applying translation generators (and their inverses).
        
        i=1
        #print(sites.size)
        while i < self.l:
            i=i+1
            N=self.sites.size
            for k in range(N):
                for n in range(N):
                    self.sites = np.append(self.sites, self.trans(self.sites[k], n)) # we apply  generators to each side...
            
            self.sites = np.unique(self.sites) # and through away the repeated ones.
 
 
        # Let us check again that no repeated sites are generated, and if they are, we through them away.
        
        i=0
        while i < self.sites.size:
            ind_to_del = np.array([], dtype=int)
            for k in range(i+1,  self.sites.size):
                if self.dist(self.sites[i], self.sites[k]) < 0.01:
                    ind_to_del=np.append(ind_to_del, k)
            self.sites = np.delete(self.sites, ind_to_del)
            i=i+1
        
        # Having generated the lattice, we can now build the adjacncy matrix.
        
        print('We have ', self.sites.size,' sites in total.')
        
        N=self.sites.size
        
        C = self.dist(r0, 0)
        B = math.asinh( math.sin( math.pi / self.p )*math.sinh(C))
        
        self.hamiltonian=np.zeros((N,N))
        
        for i in range(N):
            
            for k in range(N):
                if self.dist(self.sites[i], self.sites[k]) < 2*B+0.001:
                    if self.dist(self.sites[i], self.sites[k]) > 2*B-0.001:
                        self.hamiltonian[i, k] = -self.hopping
        
        print(self.hamiltonian)
    
    
"Class for BdG hamiltonians and all corresponding functions"
class HyperBdG():
    def __init__(self, hyperlattice, V,T,mu,Delta=[]):
        self.lattice_sample=hyperlattice    
        self.lattice_H=hyperlattice.hamiltonian
        self.N=len(hyperlattice.sites)
        self.hopping=hyperlattice.hopping
        self.V=V
        self.T=T
        self.mu=mu
        self.BdG_H=[]
       
        if len(Delta)==0:
            self.Delta=np.zeros(self.N)
            self.initial_Delta=False #it is necessary to obtain non-trivial solution of BdG equation
            print("no initial Delta for the BdG Hamiltonian")
        else:
            self.Delta=Delta
            self.initial_Delta=True
        
        #construct trivial hamiltonian
        self.construct_hamiltonian()
        spectra, vectors = eigh(self.BdG_H)
        self.spectra=spectra
        self.vectors=vectors
        
    #Fermi function
    def F(self, E):
        return 1/(np.exp((E)/self.T)+1)
    
    
    
    def construct_hamiltonian(self):
        H_Delta = diags([self.Delta], [0], shape=(self.N, self.N)).toarray()
        self.BdG_H = np.block([[self.lattice_H - self.mu*np.eye(self.N), H_Delta], [H_Delta, -self.lattice_H + self.mu*np.eye(self.N)]])
    
    
    def charge_density(self):
        fermi_dist=self.F(self.spectra[self.N:])
        v=self.vectors[self.N:,self.N:]
        u=self.vectors[:self.N,self.N:]
        n=2*np.einsum(u,[0,1],np.conj(u), [0,1],fermi_dist,[1],[0])+2*np.einsum(v,[0,1],np.conj(v), [0,1],np.ones(self.N)-fermi_dist,[1],[0])
    
        return n
    
    
    def BdG_cycle(self):
        
        print("charge density, n", np.mean(self.charge_density()))
        print("BdG cycle T=", self.T)
        step=0
        if self.initial_Delta==False:
            self.Delta=0.5*np.ones(self.N)+0.1*np.random.rand(self.N)
            self.construct_hamiltonian()
            spectra, vectors = eigh(self.BdG_H)
            self.spectra=spectra
            self.vectors=vectors
            
        while True:
            F_weight=np.ones(self.N)-2*self.F(self.spectra[self.N:])
            vectors_up=self.V * np.conj(self.vectors[self.N:,self.N:])
            Delta_next= np.einsum(vectors_up, [0,1], self.vectors[:self.N,self.N:], [0,1], F_weight,[1],[0])
            error=np.max(np.abs((self.Delta-Delta_next)))
            self.Delta=Delta_next
            print("step", step, "error", error, "Delta_max", np.max(np.abs(self.Delta)))
            step += 1
            if error<10**(-6):
                break
            self.construct_hamiltonian()
            spectra, vectors = eigh(self.BdG_H)
            self.spectra=spectra
            self.vectors=vectors
            
    def field_plot(self, field, fieldname='',title='', edges=True):
        
        def connectpoints(p1,p2):
            x1,x2=p1[0],p2[0]
            y1,y2=p1[1],p2[1]
            plt.plot([x1,x2],[y1,y2],color='grey',zorder=0)

        coords=np.zeros((self.N,2))
        coords[:,0]=np.real(self.lattice_sample.sites)
        coords[:,1]=np.imag(self.lattice_sample.sites)


        plt.rc('font', family = 'serif', serif = 'cmr10')
        rc('text', usetex=True)
        rc('axes', titlesize=40)

        print("plotting figure")
        fig, ax = plt.subplots(figsize=(12.8,9.6))
        
        if edges:
            for point_ind in range(self.N):
                neigh_inds=np.nonzero(self.lattice_H[point_ind,:])
                for neigh_ind in neigh_inds[0]:
                    connectpoints(coords[point_ind],coords[neigh_ind])

        cmp = plt.cm.get_cmap('plasma')
        sc=ax.scatter(coords[:,0], coords[:,1], s=40, c=field, cmap=cmp)
        
        cbar=fig.colorbar(sc)
        cbar.ax.tick_params(labelsize=24)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.set_label(title, fontsize=24, rotation=0, labelpad=-35, y=1.1)
        cbar.update_ticks()
        #cbar.ax.set_title(title,fontsize=28)
        ax.axis('off')
        #plt.title(title)
        figname=fieldname+"_V={}_T={}_mu={}_hyperbolic_p={}_q={}_l={}.pdf".format(self.V,self.T,self.mu, self.lattice_sample.p, self.lattice_sample.q, self.lattice_sample.l)
        plt.savefig(figname)
        plt.close()

def main():
    "Sample run input"
    
    p=8
    q=3
    l=3
    t=1
    V=5
    T=0.02
    mu=0
    hypersample=HyperLattice(p,q,l,t)
    #print(hypersample.hamiltonian)
    #print(hypersample.sites)
    BdGhypersample=HyperBdG(hypersample,V,T,mu)
    BdGhypersample.BdG_cycle()
    #print(BdGhypersample.Delta)
    BdGhypersample.field_plot(np.round(BdGhypersample.Delta,4))

main()