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
import math

class Tree_graph:
    def __init__(self,q,l,hopping):
        self.q=q
        self.p='infty'
        self.l=l
        self.hopping=hopping
        self.sites=np.array([[0,0,0]]) #[complex coordinates,relative angle, site number]
        self.a=1 #boost parameter
        self.n_sites=int(np.rint(1+q*((q-1)**l-1)/(q-2))) #number of sites
        self.hamiltonian=np.zeros((self.n_sites,self.n_sites))
        self.create_tree_graph()

    def set_sites_hoppings(self,i,j):
        ind1=int(np.real(i))
        ind2=int(np.real(j))
        self.hamiltonian[ind1, ind2] = -self.hopping
        self.hamiltonian[ind2, ind1] = -self.hopping
  
    def rot(self, z: np.complex128, phi)-> np.complex128:
        return (math.cos(phi) + math.sin(phi) * 1j) * z
        
    def trans(self, z: np.complex128, phi)-> np.complex128:
        boost = self.rot(self.a,phi)
        return z+boost
   
    # It is useful to define the distance function    
    def euсlid_dist(self, x, y):
        return np.abs(x-y)
    
    def create_tree_graph(self):
        
        site_index=0 #variable nedeed for taking care of sites numeration and vuilding hamiltonian
        #Initial points
        out_layer=np.array([]).reshape(0,3)
        for n in range(self.q):
            site_index+=1
            new_site=np.array( [[self.trans(self.sites[0,0], 2*math.pi*n/self.q), 2*math.pi*n/self.q, site_index ]])
            out_layer = np.vstack([out_layer,new_site])
            self.set_sites_hoppings(0, site_index)
            
        self.sites=np.concatenate((self.sites, out_layer),axis=0) #add new sites to already calculated ones

        i=1
        while i < self.l:
            i=i+1
            next_out_layer=np.array([]).reshape(0,3)
            for k in range(out_layer[:,0].size):
                for n in range(self.q-1):
                    site_index+=1
                    phi=2*math.pi*(n-0.5*(self.q-2))*(1/(self.q-0.6))**i+ out_layer[k,1]
                    new_site= [[self.trans(out_layer[k,0], phi), phi, site_index ]]
                    next_out_layer = np.vstack([next_out_layer,new_site])
                    self.set_sites_hoppings(out_layer[k,2], site_index)

            self.sites=np.concatenate((self.sites, next_out_layer),axis=0) #add new sites to already calculated ones
            out_layer=np.copy(next_out_layer)

        self.sites=self.sites[:,0]
        print('We have ', self.sites.size,' sites in total. Analytical result is ', self.n_sites)            
     
 
"Class for initial one-particle lattice and corresponding hamiltonian"
class HyperLattice:
    def __init__(self,p,q,l,hopping):
        self.p=p
        self.q=q
        self.l=l
        self.hopping=hopping
        self.sigma=math.sqrt( (math.cos(2*math.pi / self.p) + math.cos(2*math.pi / self.q)) / (1 + math.cos(2*math.pi / self.q)) )

        self.sites=np.array([])
        self.hamiltonian=[]
        self.create_hyperbolic_lattice()

    
    def gamma(self, z: np.complex128) -> np.complex128:
        return  (z + self.sigma) / (self.sigma * z + 1)
    
    def rot(self, z: np.complex128, n)-> np.complex128:
        return (math.cos(2 * math.pi * n / self.p) + math.sin(2 * math.pi * n / self.p) * 1j) * z
        
    def trans(self, z: np.complex128, n)-> np.complex128:
        result = self.rot(z,-n+0.5*self.p)
        result = self.gamma(result)
        result = self.rot(result, n)
        return result
    
    # It is useful to define the distance function    
    def dist(self, x, y):
        return math.acosh(1 + 2*abs(x-y)**2 / (1 - abs(x)**2) / (1-abs(y)**2))
   
        
    def create_hyperbolic_lattice(self):
    
        r0 = math.sqrt(math.cos(math.pi*(1/self.p +1/self.q) ) / math.cos(math.pi*(1/self.p - 1/self.q) ) )
    
        #Unit cell points    
        for n in range(self.p):
            delta=math.pi/self.p
            self.sites = np.append(self.sites, r0*math.cos(math.pi*2*n/self.p+delta) + r0*math.sin(math.pi*2*n/self.p+delta)*1j )
       
        # Next, we generate new cells by applying translation generators (and their inverses).
        i=1
        out_layer=np.copy(self.sites)
        while i < self.l:
            i=i+1
            next_out_layer=np.array([])
            for k in range(out_layer.size):
                for n in range(self.p):
                    next_out_layer=np.append(next_out_layer, self.trans(out_layer[k], n)) # we apply  generators to outer layer
            self.sites=np.concatenate((self.sites, next_out_layer), axis=0) #add new sites to already calculated ones
            self.sites = np.unique(self.sites) # and through away the repeated sites
            out_layer=np.copy(next_out_layer)
 
 
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
            for k in range(i+1,N):
                if self.dist(self.sites[i], self.sites[k]) < 2*B+0.001:
                    if self.dist(self.sites[i], self.sites[k]) > 2*B-0.001:
                        self.hamiltonian[i, k] = -self.hopping
                        self.hamiltonian[k, i] = -self.hopping

            
    
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
            
    def field_plot(self, field, fieldname=r'$\Delta$',title='', edges=True):
        
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
        cbar.set_label(fieldname, fontsize=28, rotation=0)
        cbar.update_ticks()
        #cbar.ax.set_title(title,fontsize=28)
        ax.axis('off')
        tstr='{},{}'.format(self.lattice_sample.p, self.lattice_sample.q)
        if self.lattice_sample.p=='infty':
            tstr='{},{}'.format('\infty', self.lattice_sample.q)
     
        title='$\{'+tstr+'\}$'+'  l='+str(self.lattice_sample.l)
        plt.title(title,fontsize=24)
        figname=fieldname+"_V={}_T={}_mu={}_hyperbolic_p={}_q={}_l={}.png".format(self.V,self.T,self.mu, self.lattice_sample.p, self.lattice_sample.q, self.lattice_sample.l)
        plt.show()
        #plt.savefig(figname)
        #plt.close()

#create general array of Delta depending on different parameters for a given sample
def calculate_hyperdiagram(lattice_sample, V_array, mu_array, T_array):
    Deltas={}
    for V in V_array:
        for mu in mu_array:
            Delta_seed=[]
            for T in T_array:
                print("calculating V=",V,"mu=",mu,"T=",T)
                if len(Delta_seed)>0 and np.max(Delta_seed)<10**(-6):
                    Deltas[(V,T,mu)]=np.zeros(lattice_sample.sites.size)
                    continue

                BdG_sample=HyperBdG(lattice_sample, V, T, mu, Delta=Delta_seed)
                BdG_sample.BdG_cycle()
                Deltas[(V,T,mu)]=BdG_sample.Delta
                Delta_seed=BdG_sample.Delta

    diagram={'lattice_sample':lattice_sample, 'V':V_array, 'mu':mu_array, 'T':T_array, 'Deltas':Deltas}
    filename="diagram_hyperbolic_p={}_q={}_l={}.pickle".format(lattice_sample.p, lattice_sample.q, lattice_sample.l)
    pickle.dump(diagram, file = open(filename, "wb"))

def load_hyperdiagram(lattice_sample, suffix="diagram"):  
    filename=suffix+"_hyperbolic_p={}_q={}_l={}.pickle".format(lattice_sample.p, lattice_sample.q, lattice_sample.l)
    try:
        diagram=pickle.load(file = open(filename, "rb"))
        return diagram
    except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
        return -1

def plot_diagram(lattice_sample, diagram):
    if len(diagram['V'])==1:
        V=diagram['V'][0]
        x=diagram['mu']
        y=diagram['T']
        Deltas=diagram['Deltas']
        z=np.zeros((len(x),len(y)))       
        for i in range(len(y)):
            for j in range(len(x)):
                Delta=Deltas[(V,y[i],x[j])]
                z[i,j]=np.mean(Delta)
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.rc('font', family = 'serif', serif = 'cmr10')
        rc('text', usetex=True)
        plt.xlabel(r'$\mu$',fontsize=24)
        plt.ylabel(r'$T$',fontsize=24)
        tstr='{},{}'.format(lattice_sample.p, lattice_sample.q)
        if lattice_sample.p=='infty':
            tstr='{},{}'.format('\infty', lattice_sample.q)
        title='$\{'+tstr+'\}$'+'  l='+str(lattice_sample.l)
        plt.title(title,fontsize=24)

        plt.imshow(z, vmin=z.min(), vmax=z.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
        cbar=plt.colorbar()
        cbar.set_label(r'$\bar \Delta$', fontsize=24, rotation=0, labelpad=-35, y=1.1)
        cbar.ax.tick_params(labelsize=24)
        cbar.update_ticks()
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        filename="diagram_hyperbolic_p={}_q={}_l={}.png".format(lattice_sample.p, lattice_sample.q, lattice_sample.l)

        plt.savefig(filename)
        plt.close()
