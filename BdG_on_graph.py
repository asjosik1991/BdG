import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags
from numpy import random
import time
import pickle
from numba import jit



"Class for initial one-particle lattice and corresponding hamiltonian"
class Lattice():
    def __init__(self,hopping,mode, size ,fractal_iter=0, alpha=0, beta=0, dis_array=None, del_array=None, pbc=False):

        if mode=="triangle":
            self.neigh=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)]
        if mode=="square":
            self.neigh=[(1,0),(-1,0),(0,1),(0,-1)]
        self.size=size #linear size
        self.mode=mode
        self.pbc=pbc #periodic boundary conditions
        self.sites=[]
        self.hopping=hopping
        self.fractal_iter=fractal_iter
        self.alpha=alpha
        self.beta=beta
        self.hamiltonian=[]
        self.dis_array=[]
        self.del_array=[]
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

        if self.mode == "triangle":
            for i in range(self.size):
                for j in range(i+1):
                    self.sites.append((i,j))

        if self.mode == "square":
            for i in range(self.size):
                for j in range(self.size):
                    self.sites.append((i,j))

    def add_disorder(self):
        N=len(self.sites)
        delete_list=[]
        for i in range(N):
            x=random.random_sample()
            if x>=(1-self.alpha):
                delete_list.append(self.sites[i])

        print("delete_array", delete_list)
        for site in delete_list:
            self.sites.remove(site)


    def Sierpinski_carpet(self):
        power=0
        counter=self.size
        while counter>1:
            size_check=counter%3
            counter=counter//3
            if size_check>0:
                break                
            power+=1
        
        if power<self.fractal_iter:
            print("the size is incompatible with fractal iterations")
            return
            
        #print("power", power)
        del_array = []
        for x in range(self.size):
            for y in range(self.size):
                for k in range(self.fractal_iter):

                    xdel = False
                    ydel = False

                    xtest = (x // (3 ** (power - (k + 1)))) % 3
                    if xtest == 1:
                        xdel = True
                    ytest = (y // (3 ** (power - (k + 1)))) % 3
                    if ytest == 1:
                        ydel = True

                    if xdel == True and ydel == True:
                        del_array.append((x,y))
                        break

        #print("del_array",del_array)
        for site in del_array:
            self.sites.remove(site)

        N = len(self.sites)
        H=np.zeros((N,N))
        test_site_set=set(self.sites)
        for n_site in range(N):
            #print("n_site, n_coord", n_site, self.sites[n_site])
            for neigh_vec in self.neigh:
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))
                #print("neigh_coord", neigh_coord)
                if neigh_coord in test_site_set:
                    n_neigh = self.sites.index(neigh_coord)
                    #print("n_neigh, neigh_coord", n_neigh, neigh_coord)
                    H[n_site,n_neigh]=-self.hopping

        self.hamiltonian = H

    def Sierpinski_gasket(self):

        power=1
        counter=self.size-1
        while counter>1:
            size_check=counter%2
            counter=counter//2
            if size_check>0:
                print("size is incompatible with Sierpinski gasket")
                return
            power+=1
        print("maximum iteration number", power)

        del_array=[]
        for x in range(self.size):
            for y in range(x+1):
                for k in range(self.fractal_iter):
                    #print("k",k)
                    xdel = False
                    ydel = False
                    diagdel=False
                    xtest = (x // (2 ** (power - (k + 2)))) % 2
                    if xtest == 1 and x%(2 ** (power - (k + 2)))>0:
                        xdel = True

                    ytest = ((2**(power-1)-y) // (2 ** (power - (k + 2)))) % 2
                    if ytest == 1 and (2**(power-1)-y)%(2 ** (power - (k + 2)))>0:
                        ydel = True

                    diagtest= ((2**(power-1)-(x-y)) // (2 ** (power - (k + 2)))) % 2
                    if diagtest == 1 and (2**(power-1)-(x-y))%(2 ** (power - (k + 2)))>0:
                        diagdel = True

                    if xdel == True and ydel == True and diagdel==True:
                        #print("total",x,y, "k=",k)
                        del_array.append((x,y))
                        break

        #print("del_array",del_array)
        for site in del_array:
            self.sites.remove(site)

        if self.alpha>0:
            self.add_disorder()

        N = len(self.sites)
        H=np.zeros((N,N))
        #print(self.sites)
        test_site_set=set(self.sites)
        l_min=2 ** (power - (self.fractal_iter + 1))#test for boundary coordinates
        print("l_min", l_min)
        for n_site in range(N):
            #print("n_site, n_coord", n_site, self.sites[n_site])

            x=self.sites[n_site][0]
            y=self.sites[n_site][1]
            #print("site", x,y)
            for neigh_vec in self.neigh:
                if self.pbc:
                    neigh_coord = tuple((a + b)%(self.size) for a, b in zip(self.sites[n_site], neigh_vec))
                else:
                    neigh_coord = tuple(a + b for a, b in zip(self.sites[n_site], neigh_vec))

                #print("neigh_coord", neigh_coord)

                if x%l_min==0 and y%l_min!=0 and (neigh_vec==(1,0) or neigh_vec==(1,1)):
                    #print("xcheck", x,y)
                    continue
                if y%l_min==0 and x%l_min!=0 and (neigh_vec==(0,-1) or neigh_vec==(-1,-1)):
                    #print("ycheck", x,y)
                    continue
                if (x-y)%l_min==0 and (y%l_min!=0 and x%l_min!=0) and (neigh_vec==(-1,0) or neigh_vec==(0,1)):
                    #print("diagcheck", x,y)
                    continue
                if neigh_coord in test_site_set:
                    n_neigh = self.sites.index(neigh_coord)
                    "edge test"
                    #print("n_neigh, neigh_coord", n_neigh, neigh_coord)
                    H[n_site,n_neigh]=-self.hopping

        self.hamiltonian = H


    def create_hamiltonian(self):

        self.create_list_of_sites()
        
        if self.mode=="triangle":
           self.Sierpinski_gasket()
           
        if self.mode=="square":
           self.Sierpinski_carpet()
           
        if self.alpha>0:
            self.add_disorder()

        if len(self.del_array)>0:
            self.del_array.sort(reverse=True)
            print("del_sort", self.del_array)
            for n in self.del_array:
                print("holes disorder", self.sites[n], "n", n)
                self.sites.remove(self.sites[n])

        if len(self.dis_array)>0:
            for n in self.dis_array:
                print("V_disorder site", self.sites[n], "n", n)
                self.hamiltonian[n,n]=100
    
    
    #figure of a lattice
    def show_graph(self):
        pos=self.sites
        plt.scatter(pos[:,0], pos[:,1], s=4)
        plt.savefig("graph.png")
        plt.close()



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
        
        if len(Delta)==0:
            self.Delta=np.zeros(self.N)
            self.initial_Delta=False #it is necessary to obtain non-trivial solution of BdG equation
        else:
            self.Delta=Delta
            self.initial_Delta=True
        
        #construct trivial hamiltonian
        self.construct_hamiltonian()
        spectra, vectors = eigh(self.BdG_H)
        self.spectra=spectra
        self.vectors=vectors
        #print("spectra", spectra)


    #Fermi function
    def F(self, E):
        return 1/(np.exp((E)/self.T)+1)
   
    #@jit(nopython=True)
    def local_kinetic_energy(self):
        
        print("Kinetic energy is being calculated")
        
        site_set=set(self.lattice_sample.sites)
        K=np.zeros(self.N, dtype=complex)
        u=self.vectors[:self.N,self.N:]
        v=self.vectors[self.N:,self.N:]
        energies=self.spectra[self.N:]
        
        u_x=np.zeros((self.N,self.N))
        v_x=np.zeros((self.N,self.N))               
        
        #prepare translated eigenvectors
        for i in range(self.N):
            coord=self.lattice_sample.sites[i]
            if self.pbc:
                coord_x= tuple((a + b)%self.size for a, b in zip(coord, (1,0)))
                #print(coord, coord_x)
            else:
                coord_x= tuple(a + b for a, b in zip(coord, (1,0)))
            if coord_x in site_set:
                i_x=self.lattice_sample.sites.index(coord_x)
                u_x[i,:]=u[i_x,:]
                v_x[i,:]=v[i_x,:]
        
        uu_x=2*self.hopping*(u*np.conj(u_x)+np.conj(u)*u_x)
        vv_x=2*self.hopping*(v*np.conj(v_x)+np.conj(v)*v_x)
        
        K=np.einsum(uu_x,[0,1],self.F(energies),[1],[0])+np.einsum(vv_x,[0,1],self.F(-energies),[1],[0])

        return K


    def twopoint_correlator(self, q_y):
        
        print("correlator is being calculated")
        site_set=set(self.lattice_sample.sites)
        # Lambda=np.zeros((len(q_y), self.N), dtype=complex)
        u=self.vectors[:self.N,:]
        v=self.vectors[self.N:,:]
        
        u_x=np.zeros((self.N,2*self.N))
        v_x=np.zeros((self.N,2*self.N))
        exp_a=np.zeros(self.N, dtype=complex)
        exp_d=np.zeros(self.N, dtype=complex)

        
        #prepare translated eigenvectors
        for i in range(self.N):
            coord=self.lattice_sample.sites[i]
            
            exp_a[i]=np.exp(-1j*q_y*coord[1])
            exp_d[i]=np.exp(1j*q_y*coord[1])

            if self.pbc:
                coord_x= tuple((a + b)%self.size for a, b in zip(coord, (1,0)))
                #print(coord, coord_x)
            else:
                coord_x= tuple(a + b for a, b in zip(coord, (1,0)))
            if coord_x in site_set:
                i_x=self.lattice_sample.sites.index(coord_x)
                u_x[i,:]=u[i_x,:]
                v_x[i,:]=v[i_x,:]


        # uu_x=np.zeros((self.N,2*self.N,2*self.N))
        # vv_x=np.zeros((self.N,2*self.N,2*self.N))
        # for i in range(self.N): 
        #     uu_x[i,...]=np.einsum(np.conj(u[i,:]), [1], u_x[i,:], [2], [1,2])
        #     vv_x[i,...]=np.einsum(np.conj(v[i,:]), [1], v_x[i,:], [2], [1,2])
        
        #prepare fermi weight
        
        # F_weight_test=np.zeros((2*self.N,2*self.N))
        # for i in range(2*self.N):
        #     for j in range(2*self.N):
        #         F_weight_test[i,j]=(self.F(self.spectra[i])-self.F(self.spectra[j]))/(self.spectra[i]-self.spectra[j]+1j*10**(-6))
        
        energy_diff= np.tile(self.spectra, (2*self.N,1)) - np.tile(self.spectra, (2*self.N,1)).T+1j*10**(-6)
        fermi_diff= np.tile(self.F(self.spectra), (2*self.N,1)) - np.tile(self.F(self.spectra), (2*self.N,1)).T
        F_weight=fermi_diff/energy_diff
        
        # print("F_test", np.max(np.abs(F_weight-F_weight_test)))
        
        


        uu_x=np.einsum(exp_a, [0], np.conj(u_x), [0,1], u, [0,2], [1,2])
        uu_x_t=np.einsum(exp_a, [0], np.conj(u), [0,1], u_x, [0,2], [1,2])

        # vv_x=np.einsum(exp_d, [0], v_x, [0,1], np.conj(v), [0,2], [1,2])
        # vv_x_t=np.einsum(exp_d, [0], v, [0,1], np.conj(v_x), [0,2], [1,2])
        
        A=uu_x-uu_x_t
        # D=vv_x-vv_x_t
        
        Lambda=1/self.N*np.einsum(A, [0,1], np.conj(A), [0,1], F_weight, [0,1])
        
        #test
        # k=np.linspace(0, 2*np.pi*(1-1/self.size), self.size)
        
        # K_diff= np.tile(self.spectra, (2*self.N,1)) - np.tile(self.spectra, (2*self.N,1)).T+1j*10**(-6)


        

 
        # u_c=np.conj(u)
        # v_c=np.conj(v)

        # for i in range(self.N):
        #     print('i site', i)
        #     for j in range(self.N):
        #         for n in range(self.N):
        #             for m in range(self.N):
        #                 coord_i=self.lattice_sample.sites[i]
        #                 coord_ix= tuple(a + b for a, b in zip(coord_i, (1,0)))
        #                 coord_j=self.lattice_sample.sites[i]
        #                 coord_jx= tuple(a + b for a, b in zip(coord_j, (1,0)))
        #                 if (coord_ix in site_set) and (coord_jx in site_set):
        #                     i_x= self.lattice_sample.sites.index(coord_ix)
        #                     j_x= self.lattice_sample.sites.index(coord_jx)
    
        #                     a=u[j,n]*u_c[i_x,n]*u[i,m]*u_c[j_x,m] - u_c[i_x,n]*v[j_x,n]*u[i,m]*v_c[j,m] - u[j_x,n]*u_c[i_x,n]*u[i,m]*u_c[j,m]\
        #                                +u_c[i_x,n]*v[j,n]*u[i,m]*v_c[j_x,m] - u[j,n]*u_c[i,n]*u[i_x,m]*u_c[j_x,m]+ u_c[i,n]*v[j_x,n]*u[i_x,m]*v_c[j,m]\
        #                                + u[j_x,n]*u_c[i,n]*u[i_x,m]*u_c[j,m] - u_c[i,n]*v[j,n]*u[i_x,m]*v_c[j_x,m]
    
        #                     b=u[j,n]*u_c[i_x,n]*v_c[i,m]*v[j_x,m] + u_c[i_x,n]*v[j_x,n]*v_c[i,m]*u[j,m] - u[j_x,n]*u_c[i_x,n]*v_c[i,m]*v[j,m]\
        #                                -u_c[i_x,n]*v[j,n]*v_c[i,m]*u[j_x,m] - u[j,n]*u_c[i,n]*v_c[i_x,m]*v[j_x,m]- u_c[i,n]*v[j_x,n]*v_c[i_x,m]*u[j,m]\
        #                                + u[j_x,n]*u_c[i,n]*v_c[i_x,m]*v[j,m] + u_c[i,n]*v[j,n]*v_c[i_x,m]*u[j_x,m]
    
        #                     c =v_c[j,n]*v[i_x,n]*u[i,m]*u_c[j_x,m] + v[i_x,n]*u_c[j_x,n]*u[i,m]*v_c[j,m] - v_c[j_x,n]*v[i_x,n]*u[i,m]*u_c[j,m]\
        #                                -v[i_x,n]*u_c[j,n]*u[i,m]*v_c[j_x,m] - v_c[j,n]*v[i,n]*u[i_x,m]*u_c[j_x,m]- v[i,n]*u_c[j_x,n]*u[i_x,m]*v_c[j,m]\
        #                                + v_c[j_x,n]*v[i,n]*u[i_x,m]*u_c[j,m] + v[i,n]*u_c[j,n]*u[i_x,m]*v_c[j_x,m]
    
        #                     d = v_c[j,n]*v[i_x,n]*v_c[i,m]*v[j_x,m] - v[i_x,n]*u_c[j_x,n]*v_c[i,m]*u[j,m] - v_c[j_x,n]*v[i_x,n]*v_c[i,m]*v[j,m]\
        #                                +v[i_x,n]*u_c[j,n]*v_c[i,m]*u[j_x,m] - v_c[j,n]*v[i,n]*v_c[i_x,m]*v[j_x,m]+ v[i,n]*u_c[j_x,n]*v_c[i_x,m]*u[j,m]\
        #                                + v_c[j_x,n]*v[i,n]*v_c[i_x,m]*v[j,m] - v[i,n]*u_c[j,n]*v_c[i_x,m]*u[j_x,m]
    
        #                     Lambda[:,i]+=self.hopping**2/self.N*np.exp(1j*q_y*(coord_i[1]-coord_j[1]))*((a+d)*(self.F(energies[n])-self.F(energies[m]))/(1j*10**(-6)+energies[n]-energies[m])
        #                                                                 +(b+c)*(self.F(energies[n])+self.F(energies[m]))/(1j*10**(-6)+energies[n]+energies[m]))
        
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
        #print(energies)
        v=self.vectors[self.N:,self.N:]
        u=self.vectors[:self.N,self.N:]
        n=2*np.einsum(u,[0,1],np.conj(u), [0,1],fermi_dist,[1],[0])+2*np.einsum(v,[0,1],np.conj(v), [0,1],np.ones(self.N)-fermi_dist,[1],[0])
    
        return n
    
    
    def BdG_cycle(self):
        
        print("BdG cycle T=", self.T)
        step=0
        if self.initial_Delta==False:
            self.Delta=0.1*np.ones(self.N)+0.1*np.random.rand(self.N)
            self.construct_hamiltonian()
            spectra, vectors = eigh(self.BdG_H)
            self.spectra=spectra
            self.vectors=vectors
            
        while True:
            vectors_up=self.V * np.conj(self.vectors[self.N:,self.N:])
            Delta_next= np.einsum(vectors_up, [0,1], self.vectors[:self.N,self.N:], [0,1], np.ones(self.N)-2*self.F(self.spectra[self.N:]),[1],[0])
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
        
        print("charge density, n", np.mean(self.charge_density()))


#Fermi function
def F(epsilon,T):
    return 1/(np.exp(epsilon/T)+1)

#analytical formula for uniform case, for tests
def uniform_2D_correlation_function(size, T, Delta=0, state='normal'):
    
    N_qy=100
    k=np.linspace(0, 2*np.pi*(1-1/size), size)
    q_y=np.linspace(2*np.pi/N_qy, 2*np.pi*(1-1/N_qy), N_qy)
    Lambda=np.zeros(N_qy)
    
    if state=='normal':
        
        for i in range(N_qy):
            for k_x in k:
                for k_y in k:
                    eps=-2*(np.cos(k_x)+np.cos(k_y))
                    eps_qy=-2*(np.cos(k_x)+np.cos(k_y+q_y[i]))
                    #print(eps, eps_qy)
                    Lambda[i]+=np.real(-8/(size**2)*(np.sin(k_x)**2)*(F(eps,T)-F(eps_qy,T))/(eps-eps_qy+1j*10**(-6)))
            
        #kinetic energy from limit
        K_simple=0
        K_withderivative=0
        for k_x in k:
            for k_y in k:
                
                eps=-2*(np.cos(k_x)+np.cos(k_y))
                # print('kx, ky, eps', np.round(k_x/np.pi, 2), np.round(k_y/np.pi, 2), np.round(eps,2))
                K_simple+=4*np.cos(k_x)*F(eps,T)/(size**2)
                K_withderivative+=8*(np.sin(k_x)**2)/(4*T*(size**2)*(np.cosh(eps/(2*T))**2))

        print("analytic kinetic energy", K_simple, "\n analytical limit", K_withderivative, "\n numerical limit", Lambda[0])
        
        return q_y, Lambda
    
    if state=='super':
        
        for i in range(N_qy):
            #print("q_y", q_y[i])
            for k_x in k:
                for k_y in k:
                    eps=-2*(np.cos(k_x)+np.cos(k_y))
                    eps_qy=-2*(np.cos(k_x)+np.cos(k_y+q_y[i]))
                    E=np.sqrt(eps**2 + Delta**2)
                    E_qy=np.sqrt(eps_qy**2 + Delta**2)
                    L=0.5*(1 + (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    P=0.5*(1- (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    #print(np.round(L,4), np.round(P,4), E, E_qy)
                    Lambda[i]+=4/(size**2)*(np.sin(k_x)**2)*(L*(1/(1j*10**(-6)+E-E_qy)+1/(-1j*10**(-6)+E-E_qy)*(F(E,T)-F(E_qy,T)))
                                                             +P*(1/(1j*10**(-6)+E+E_qy)+1/(-1j*10**(-6)+E+E_qy)*(1-F(E,T)-F(E_qy,T))))
                    
                    #Lambda[i]+=np.real(-8/(size**2)*(np.sin(k_x)**2)*(F(E,T)-F(E_qy,T))/(E-E_qy+1j*10**(-6)))

        return q_y, Lambda
    

#create \Delta-T diagram for a given sample
def T_diagram(lattice_sample, V, mu, T_array):
    Delta_array=[]
    Delta_ini=[]
    for T in T_array:
        BdG_sample=BdG(lattice_sample, V, T, mu, Delta=Delta_ini)
        BdG_sample.BdG_cycle()
        Delta_ini=BdG_sample.Delta     
        Delta_array.append(BdG_sample.Delta)
    
    T_diagram_obj={'lattice_sample':lattice_sample, 'V':V, 'mu':mu, 'T_array':T_array, 'Delta_array':Delta_array}
    filename="T_diagram_V={}_mode={}_fractiter={}.pickle".format(V, lattice_sample.mode, lattice_sample.fractal_iter)
    pickle.dump(T_diagram_obj, file = open(filename, "wb"))
    plot_T_diagram(T_diagram_obj)


#plot \Delta-T diagram for a given sample
def plot_T_diagram(T_diagram_obj):
    
    Nt=len(T_diagram_obj['T_array'])
    V=T_diagram_obj['V']
    T_array=np.array(T_diagram_obj['T_array'])
    Delta_array=np.array(T_diagram_obj['Delta_array'])
    lattice_sample=T_diagram_obj['lattice_sample']
    
    #print(Delta_array)
    Delta_av=np.zeros(Nt)
    for i in range(Nt):
        Delta_av[i]=np.mean(Delta_array[i,:])
        
    plt.plot(T_array, Delta_av)

    title="V={} mode='{}' fractiter={}".format(V, lattice_sample.mode, lattice_sample.fractal_iter)
    plt.xlabel("T")
    plt.ylabel(r'$<\Delta>$')
    plt.title(title)

    figname="T_diagram_V={}_mode={}_fractiter={}.png".format(V, lattice_sample.mode, lattice_sample.fractal_iter)
    plt.savefig(figname)
    plt.close()


def main():

    mode="square"
    t=1
    size=9
    T=1
    V=0.0
    mu=0
    
    lattice_sample = Lattice(t, mode, size, fractal_iter=0, pbc=True)
    BdG_sample=BdG(lattice_sample, V, T, mu)
    #BdG_sample.BdG_cycle()
    # time1=time.time()
    # K=BdG_sample.local_kinetic_energy()
    # print("K", K)
    # print("mean K", np.mean(K))
    # time2=time.time()
    # print("kinenergy time", time2-time1)
    #print("spectra", BdG_sample.spectra)
    # print("kinetic energy time", time2-time1)
    
    # q_y=2*np.pi/100
    # Lambda=BdG_sample.twopoint_correlator(q_y)
    # print("lambda", Lambda)
    
    N_qy=100
    q_y=np.linspace(2*np.pi/size, 2*np.pi*(1-1/size), N_qy)
    Lambda=[]
    i=0
    for q in q_y:
        print("i", i, "q", q)
        Lambda.append(BdG_sample.twopoint_correlator(q))
        print("Lambda", Lambda[i])
        i+=1
    plt.plot(q_y, Lambda)
    plt.savefig("lambda_numerical_test.png")
    plt.close()
    
    # BdG_sample.BdG_cycle()
    # print(BdG_sample.spectra)
    # spectra, vectors = eigh(lattice_sample.hamiltonian)
    #print(spectra)
    
    
    q_y, Lambda=uniform_2D_correlation_function(size, T, Delta=0, state='normal')
    # print("zero limit", Lambda[0])
    plt.plot(q_y, Lambda)
    plt.savefig("lambda_test_normal.png")
    plt.close()
    
 
    # q_y, Lambda=uniform_2D_correlation_function(size, T, Delta=0.0, state='super')
    # print("zero limit", Lambda[0])
    # plt.plot(q_y, Lambda)
    # plt.savefig("lambda_test_super.png")
    # plt.close()
    
    #n=BdG_sample.charge_density()
    #print("charge density", np.sum(n)/len(lattice_sample.sites))
    #K=BdG_sample.local_kinetic_energy()
    #print("kinetic energy", np.sum(K)/len(lattice_sample.sites))
    
    # lattice_sample = Lattice(t, mode, size, fractal_iter=0, pbc=True)
    # for v in [V]:
    #     V=round(v,2)
    #     T_array=np.linspace(0.01, 0.25, num=100)
    #     T_diagram(lattice_sample, V, mu, T_array)



main()