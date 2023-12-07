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

"Class for initial one-particle lattice and corresponding hamiltonian"
class Lattice():
    def __init__(self,hopping,mode, size ,fractal_iter=0, alpha=0, beta=0, dis_array=None, del_array=None, pbc=True, noise=True):

        if mode=="triangle" or "regular_triangle":
            self.neigh=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]
        if mode=="square":
            self.neigh=[(1,0),(-1,0),(0,1),(0,-1)]
        if mode=="1dchain":
            self.neigh=[(1,0),(-1,0)]
        if mode=="triangle_lattice":
            self.neigh=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]

        self.size=size #linear size
        self.mode=mode
        self.pbc=pbc #periodic boundary conditions
        self.sites=[]
        self.hopping=hopping
        self.fractal_iter=fractal_iter
        self.alpha=alpha
        self.noise=noise
        self.beta=beta
        self.hamiltonian=[]
        self.dis_array=[]
        self.del_array=[]
        self.sites_number=0
        if dis_array==None:
            self.dis_array=[]
        else:
            self.dis_array=dis_array
        if del_array==None:
            self.del_array=[]
        else:
            self.del_array=del_array
        #print("Hamiltonian parameteres: mode", self.mode, "dis_array", self.dis_array, "del_array", self.del_array)
        self.create_hamiltonian()
        
    def create_list_of_sites(self):
        
        if self.mode=="1dchain":
            for i in range(self.size):
                self.sites.append((i,0))

        if self.mode == "triangle" or "regular_triangle":
            for i in range(self.size):
                for j in range(i+1):
                    self.sites.append((i,j))

        if self.mode == "square":
            for i in range(self.size):
                for j in range(self.size):
                    self.sites.append((i,j))
        
        if self.mode == "triangle_lattice":
            for i in range(self.size):
                for j in range(self.size):
                    self.sites.append((i,j))

    def add_disorder(self):
        
        if self.alpha>0 and len(self.del_array)==0:
            N=len(self.sites)
            for i in range(N):
                x=random.random_sample()
                if x>=(1-self.alpha):
                    self.del_array.append(i)

        if len(self.del_array)>0:
            self.del_array.sort(reverse=True)
            print("del_sort", self.del_array)
            for n in self.del_array:
                print("holes disorder", self.sites[n], "n", n)
                self.sites.remove(self.sites[n])
        
        N = len(self.sites)
        self.hamiltonian=np.zeros((N,N))
        self.sites_number=N

        if len(self.dis_array)>0:
            for n in self.dis_array:
                print("V_disorder site", self.sites[n], "n", n)
                self.hamiltonian[n,n]=5
        
        #noise is important, because it regularize degenerate eigenvalues. otherwise, kinetic energy becomes incorrect for some hamiltonians
        if self.noise:
            for i in range(self.sites_number):
                self.hamiltonian[i,i]=10**(-5)*(random.random_sample()-0.5)
            

    def Sierpinski_carpet(self):
        power=0
        counter=self.size
        while counter>1:
            size_check=counter%3
            counter=counter//3
            if size_check>0:
                break                
            power+=1
        l=self.size//3**(power)#normalization factor for lengths not equal to powers of 3
        print("power", power, "l", l)
        if power<self.fractal_iter:
             print("the chozen fractal iterations and size don't correspond to exact fractal or number of iterations is too large")
            
        del_array = []
        for x in range(self.size):
            for y in range(self.size):
                for k in range(self.fractal_iter):

                    xdel = False
                    ydel = False
                    
                    xtest = (x// (l*3 ** (power - (k + 1)))) % 3
                    if xtest == 1:
                        xdel = True
                    ytest = (y // (l*3 ** (power - (k + 1)))) % 3
                    if ytest == 1:
                        ydel = True

                    if xdel == True and ydel == True:
                        del_array.append((x,y))
                        break
                        
        for site in del_array:
            self.sites.remove(site)
        self.add_disorder()

        test_site_set=set(self.sites)
        for n_site in range(self.sites_number):
            for neigh_vec in self.neigh:
                
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))
                    
                if neigh_coord in test_site_set:
                    
                    n_neigh = self.sites.index(neigh_coord)
                    self.hamiltonian[n_site,n_neigh]=-self.hopping


    def Sierpinski_gasket(self):

        power=1
        counter=self.size-1
        while counter>1:
            size_check=counter%2
            counter=counter//2
            if size_check>0:
                break
            power+=1
        

        l=(self.size-1)//2**(power-1)    
        if l>1:
            if power-1<self.fractal_iter:
                print("the chozen fractal iterations and size don't correspond to exact fractal or number of iterations is too large")
        if l==1:
            if power-2<self.fractal_iter:
                print("number of iterations is too large for the chosen size. the lattice will be calculated for the maximum number of fractal iterations")

        del_array=[]
        for x in range(self.size):
            for y in range(x+1):
                for k in range(self.fractal_iter):
                    xdel = False
                    ydel = False
                    diagdel=False
                    xtest = (x // (l*2 ** (power - (k + 2)))) % 2
                    if xtest == 1 and x%(l*2 ** (power - (k + 2)))>0:
                        xdel = True

                    ytest = ((l*2**(power-1)-y) // (l*2 ** (power - (k + 2)))) % 2
                    if ytest == 1 and (l*2**(power-1)-y)%(l*2 ** (power - (k + 2)))>0:
                        ydel = True

                    diagtest= ((l*2**(power-1)-(x-y)) // (l*2 ** (power - (k + 2)))) % 2
                    if diagtest == 1 and (l*2**(power-1)-(x-y))%(l*2 ** (power - (k + 2)))>0:
                        diagdel = True

                    if xdel == True and ydel == True and diagdel==True:
                        del_array.append((x,y))
                        break

        
             
        for site in del_array:
            self.sites.remove(site)
        self.add_disorder()


        test_site_set=set(self.sites)
        l_min=np.max([l*2 ** (power - (self.fractal_iter + 1)),l*2])#test for boundary coordinates, if fractal iterations are too large, it is fixed on its maximum value

        for n_site in range(self.sites_number):
            x=self.sites[n_site][0]
            y=self.sites[n_site][1]
            for neigh_vec in self.neigh:
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))

                if x%l_min==0 and y%l_min!=0 and (neigh_vec==(1,0) or neigh_vec==(1,1)):
                    continue
                if y%l_min==0 and x%l_min!=0 and (neigh_vec==(0,-1) or neigh_vec==(-1,-1)):
                    continue

                if (x-y)%l_min==0 and (y%l_min!=0 and x%l_min!=0) and (neigh_vec==(-1,0) or neigh_vec==(0,1)):
                    continue
                
                if neigh_coord in test_site_set:
                    n_neigh = self.sites.index(neigh_coord)
                    self.hamiltonian[n_site,n_neigh]=-self.hopping

        
    def onedchain(self):
        
        self.add_disorder()

        test_site_set=set(self.sites)
        for n_site in range(self.sites_number):
            for neigh_vec in self.neigh:
                
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))
                    
                if neigh_coord in test_site_set:
                    
                    n_neigh = self.sites.index(neigh_coord)
                    self.hamiltonian[n_site,n_neigh]=-self.hopping

    def triangle_lattice(self):
        self.add_disorder()
        test_site_set=set(self.sites)
        for n_site in range(self.sites_number):
            x=self.sites[n_site][0]
            y=self.sites[n_site][1]
            for neigh_vec in self.neigh:
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))                
                if neigh_coord in test_site_set:
                    n_neigh = self.sites.index(neigh_coord)
                    self.hamiltonian[n_site,n_neigh]=-self.hopping


    def triangle_transform_coordinates(self):
        a=0.5
        b=np.round(0.5*np.sqrt(3),4)
        #b=2
        self.neigh=[(1,0),(-1,0),(a,b),(-a,b),(a,-b),(-a,-b)]
        new_sites=[]
        for i in range(len(self.sites)):
            x=self.sites[i][0]
            y=self.sites[i][1]
            new_sites.append((np.round(a*y+(x-y),4),np.round(y*b,4)))
        self.sites=new_sites

    def create_hamiltonian(self):

        self.create_list_of_sites()
        
        if self.mode=="triangle":
           self.Sierpinski_gasket()
           
        if self.mode=="square":
           self.Sierpinski_carpet()
        
        if self.mode=="1dchain":
           self.onedchain()

        if self.mode=="triangle_lattice":
           self.triangle_lattice()
        
        if self.mode=="regular_triangle":
           self.Sierpinski_gasket()
           self.triangle_transform_coordinates()
    
    
    #figure of a lattice
    def show_graph(self):
        
        pos=self.sites
        G = nx.from_numpy_matrix(np.abs(self.hamiltonian))
        pos_dict={}
        for i in range(self.sites_number):
            pos_dict[i]=np.asarray(pos[i])
        nx.draw(G,pos_dict)
        plt.show()

"Class for BdG hamiltonians and all corresponding functions"
class BdG():
    def __init__(self, lattice_sample, V, T, mu, Delta=[]):
        self.lattice_sample=lattice_sample
        self.size=lattice_sample.size
        self.pbc=lattice_sample.pbc
        self.lattice_H=lattice_sample.hamiltonian
        self.N=len(lattice_sample.sites)
        self.hopping=lattice_sample.hopping
        self.V=V
        self.T=T
        self.mu=mu
        self.BdG_H=[]
        if lattice_sample.mode=="triangle":
            self.right_neighs=list(set([(1,0),(1,1)]) & set(self.lattice_sample.neigh))
        if lattice_sample.mode=="regular_triangle":
            a=0.5
            b=np.round(0.5*np.sqrt(3),4)
            self.right_neighs=[(1,0),(a,b),(a,-b)]
            
        
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
    
    def dF(self, E):
        return -1/(self.T*(np.exp(0.5*E/self.T)+np.exp(-0.5*E/self.T))**2)
    
    def local_kinetic_energy(self):
        
        print("Kinetic energy is being calculated")
        
        site_set=set(self.lattice_sample.sites)
        K=np.zeros(self.N, dtype=complex)
        u=np.copy(self.vectors[:self.N,:])
        v=np.copy(self.vectors[self.N:,:])
        energies=self.spectra
        
        K=0
        #prepare translated eigenvectors
        for neighs in self.right_neighs:
            u_x=np.zeros((self.N,2*self.N))
            v_x=np.zeros((self.N,2*self.N))      
            for i in range(self.N):
                coord=self.lattice_sample.sites[i]
                if self.pbc:
                    coord_x= tuple((a + b)%self.size for a, b in zip(coord, neighs))
                else:
                    coord_x= tuple(np.round(a + b,4) for a, b in zip(coord, neighs))
                if coord_x in site_set:
                    i_x=self.lattice_sample.sites.index(coord_x)
                    if np.abs(self.lattice_H[i,i_x])>0.01:
                        u_x[i,:]=u[i_x,:]
                        v_x[i,:]=v[i_x,:]
            dx2=neighs[0]**2
            uu_x=dx2*self.hopping*(u*np.conj(u_x)+np.conj(u)*u_x)
            vv_x=dx2*self.hopping*(v*np.conj(v_x)+np.conj(v)*v_x)
        
            K+=np.einsum(uu_x,[0,1],self.F(energies),[1],[0])+np.einsum(vv_x,[0,1],self.F(-energies),[1],[0])
        
        print("average kinetic energy", np.mean(K))
        return K

    def twopoint_correlator(self, q_y):
        #return local two-point correlation function, global value is the mean of local ones
        
        print("correlator is being calculated")
        site_set=set(self.lattice_sample.sites)
        u=self.vectors[:self.N,:]
        v=self.vectors[self.N:,:]
        

        exp_a=np.zeros(self.N, dtype=complex)
        exp_d=np.zeros(self.N, dtype=complex)
        Lambda=np.zeros(self.N, dtype=complex)

        energy_diff= np.tile(self.spectra, (2*self.N,1)) - np.tile(self.spectra, (2*self.N,1)).T
        fermi_diff= np.tile(self.F(self.spectra), (2*self.N,1)) - np.tile(self.F(self.spectra), (2*self.N,1)).T
        F_weight=np.zeros((2*self.N,2*self.N),dtype=complex) #2 is from spin indices    
        for i in range(2*self.N):
             for j in range(2*self.N):
                if np.abs(self.spectra[i]-self.spectra[j])<10**(-10):
                    F_weight[i,j]=2*self.dF(self.spectra[i])
                else:
                    F_weight[i,j]=2*fermi_diff[i,j]/energy_diff[i,j]         
        
        AD=0
        #prepare translated eigenvectors
        for neighs in self.right_neighs:

            u_x=np.zeros((self.N,2*self.N))
            v_x=np.zeros((self.N,2*self.N))
            for i in range(self.N):
                coord=self.lattice_sample.sites[i]
            
                exp_a[i]=np.exp(-1j*q_y*coord[1])
                exp_d[i]=np.exp(1j*q_y*coord[1])

                if self.pbc:
                    coord_x= tuple((a + b)%self.size for a, b in zip(coord, neighs))
                else:
                    coord_x= tuple(np.round(a + b,4) for a, b in zip(coord, neighs))
                
                if coord_x in site_set:
                    i_x=self.lattice_sample.sites.index(coord_x)
                    if np.abs(self.lattice_H[i,i_x])>0.01:
                        u_x[i,:]=u[i_x,:]
                        v_x[i,:]=v[i_x,:]
            dx=neighs[0]
            uu_x=np.einsum(exp_d, [0], u_x, [0,1], np.conj(u), [0,2], [1,2])   
            uu_x_t=np.einsum(exp_d, [0], u, [0,1], np.conj(u_x), [0,2], [1,2])       

            vv_x=np.einsum(exp_d, [0], v_x, [0,1], np.conj(v), [0,2], [1,2])  
            vv_x_t=np.einsum(exp_d, [0], v, [0,1], np.conj(v_x), [0,2], [1,2])        

            AD+=dx*(uu_x-uu_x_t+vv_x-vv_x_t)
        
        for i in range(self.N):
            a=0
            print("site ", i, "out of", self.N)
            for neighs in self.right_neighs:
                u_x=np.zeros((self.N,2*self.N))
                v_x=np.zeros((self.N,2*self.N))
                for j in range(self.N):
                    coord=self.lattice_sample.sites[j]
                    if self.pbc:
                        coord_x= tuple((a + b)%self.size for a, b in zip(coord, neighs))
                    else:
                        coord_x= tuple(np.round(a + b,4) for a, b in zip(coord, neighs))
                    if coord_x in site_set:
                        j_x=self.lattice_sample.sites.index(coord_x)
                        if np.abs(self.lattice_H[j,j_x])>0.01:
                            u_x[j,:]=u[j_x,:]
                            v_x[j,:]=v[j_x,:]
                dx=neighs[0]
                au_x=np.einsum(exp_a[i]*np.conj(u_x[i,:]), [0], u[i,:], [1], [0,1])
                au_x_t=np.einsum(exp_a[i]*np.conj(u[i,:]), [0], u_x[i,:], [1], [0,1])
                a+=dx*(au_x-au_x_t)
            
            Lambda[i]=np.einsum(a,[1,0],AD,[0,1],F_weight,[0,1])
        
        return Lambda
    
    def local_stiffness(self, q_y):
        
        K=self.local_kinetic_energy()
        Lambda=self.twopoint_correlator(q_y)
        return K-Lambda
    
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
    
    #plot a field defined on a lattice (such as order parameter or superfluid density)
    def field_plot(self, field, fieldname='',title='', edges=False,contrast=False, removeisols=True):
        
        def connectpoints(p1,p2):
            x1,x2=p1[0],p2[0]
            y1,y2=p1[1],p2[1]
            plt.plot([x1,x2],[y1,y2],color='grey',zorder=0)


        plt.rc('font', family = 'serif', serif = 'cmr10')

        rc('text', usetex=True)
        rc('axes', titlesize=40)

        print("plotting figure")
        coord=self.lattice_sample.sites
        x_coord = map(lambda x: x[0], coord)
        y_coord = map(lambda x: x[1], coord)
        x_coord=list(x_coord)
        y_coord=list(y_coord)
        viridis = cm.get_cmap('Blues', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        contrast_color = np.array(np.asarray(mcolors.to_rgb('crimson')+(1,)))
        newcolors[:1, :] =contrast_color
        newcmp = ListedColormap(newcolors)
        fig, ax = plt.subplots(figsize=(12.8,9.6))
        
        if edges:
            for point in self.lattice_sample.sites:
                for neigh_vec in self.lattice_sample.neigh:
                    neigh_point = tuple(np.round(a + b,4) for a, b in zip(point, neigh_vec))
                    if neigh_point in self.lattice_sample.sites and self.lattice_sample.hamiltonian[self.lattice_sample.sites.index(point),self.lattice_sample.sites.index(neigh_point)]!=0:
                        connectpoints(point,neigh_point)

        if removeisols:
            isols_inds=[]
            for point in self.lattice_sample.sites:
                check=False
                for neigh_vec in self.lattice_sample.neigh:
                    neigh_point = tuple(a + b for a, b in zip(point, neigh_vec))
                    if neigh_point in self.lattice_sample.sites and self.lattice_sample.hamiltonian[self.lattice_sample.sites.index(point),self.lattice_sample.sites.index(neigh_point)]!=0:
                        check=True
                if check==False:
                    isols_inds.append(self.lattice_sample.sites.index(point))
            #print("isolated indices", isols_inds)
            for index in sorted(isols_inds, reverse=True):
                #print(index)
                field=np.delete(field, index)
                del x_coord[index]
                del y_coord[index]

        if contrast:
            field_copy = np.copy(field)
            field_copy[field_copy <= 0.01] = 0.01
            sc = ax.scatter(x_coord, y_coord, s=48, c=field_copy, cmap=newcmp, zorder=1)
            #cmax=np.max([0.04, np.mean(field)+1.2*np.std(field)])
            #sc.set_clim(0, cmax)
        else:
            cmp = plt.cm.get_cmap('plasma')
            fig, ax = plt.subplots()
            sc=ax.scatter(x_coord, y_coord, s=48, c=field, cmap=cmp)
        

        
        cbar=fig.colorbar(sc)
        cbar.ax.tick_params(labelsize=40)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cbar.locator = tick_locator
        cbar.set_label(title, fontsize=48, rotation=0, labelpad=-35, y=1.1)
        cbar.update_ticks()
        #cbar.ax.set_title(title,fontsize=28)
        ax.axis('off')
        #plt.title(title)
        figname=fieldname+"_V={}_T={}_mu={}_mode={}_fractiter={}_delholes={}.pdf".format(self.V,self.T,self.mu, self.lattice_sample.mode, self.lattice_sample.fractal_iter, self.lattice_sample.alpha)
        plt.savefig(figname)
        plt.close()
        

    def plot_spectrum(self):
        
        plt.hist(self.spectra,bins=100)
        plt.show()
        plt.savefig("spectrum.png")
        plt.close()

"Calculations for different parameters"

#create general array of Delta depending on different parameters for a given sample
def calculate_diagram(lattice_sample, V_array, mu_array, T_array):
    Deltas={}
    del_arrays={}
    for V in V_array:
        for mu in mu_array:
            for T in T_array:
                print("calculating V=",V,"mu=",mu,"T=",T)
                BdG_sample=BdG(lattice_sample, V, T, mu)
                BdG_sample.BdG_cycle()
                Deltas[(V,T,mu)]=BdG_sample.Delta
                del_arrays[(V,T,mu)]=BdG_sample.lattice_sample.del_array

    diagram={'lattice_sample':lattice_sample, 'V':V_array, 'mu':mu_array, 'T':T_array, 'Deltas':Deltas, 'dels':del_arrays}
    filename="diagram_mode={}_size={}_fractiter={}_delholes={}.pickle".format(lattice_sample.mode, lattice_sample.size,lattice_sample.fractal_iter, round(lattice_sample.alpha,2))
    pickle.dump(diagram, file = open(filename, "wb"))

def load_diagram(lattice_sample, suffix="diagram"):  
    filename=suffix+"_mode={}_size={}_fractiter={}_delholes={}.pickle".format(lattice_sample.mode, lattice_sample.size, lattice_sample.fractal_iter, round(lattice_sample.alpha,2))
    try:
        diagram=pickle.load(file = open(filename, "rb"))
        return diagram
    except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
        return -1
