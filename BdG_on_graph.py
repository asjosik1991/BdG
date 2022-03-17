import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse import diags, kron
from numpy import random
import glob
import os
import networkx as nx

"systems"

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

    def create_list_of_sites(self):
        n_site=0
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
                print("size is incompatible with Sierpinski cerpet")
                return
            power+=1

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
                    H[n_site,n_neigh]=self.hopping

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
            #site_coord=
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
                    H[n_site,n_neigh]=self.hopping

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

#Fermi function
def F(E,mu,T):
    return 1/(np.exp((E-mu)/T)-1)

def local_kinetic_energy(sample, H_2p, hopping, mu,T):
    N=len(sample.sites)
    spectra, vectors = eigh(H_2p)
    site_set=set(sample.sites)
    K=np.zeros(N)

    for i in range(N): #cycle over indices
        for n in range(N):
            coord=sample.sites[i]
            coord_x= tuple(a + b for a, b in zip(coord, (1,0)))
            if coord_x in site_set:
                i_x= sample.sites.index(coord_x)
                K[i]=hopping*((np.conj(vectors[i_x,n])*vectors[i,n]+np.conj(vectors[i,n])*vectors[i_x,n])*F(spectra[N+n],mu,T)+
                        (np.conj(vectors[i_x,N+n])*vectors[i,N+n]+np.conj(vectors[i,N+n])*vectors[i_x,N+n])*(1-F(spectra[N+n],mu,T)))
    #K=K/N
    return K


def twopoint_correlator(sample, H_2p, hopping, mu,T, q_y):

    N=len(sample.sites)
    Lambda_part=np.zeros((N,N,N,N), dtype=complex)
    spectra, vectors = eigh(H_2p)
    u=vectors[:N,:]
    v=vectors[N:,:]
    u_c=np.conj(u)
    v_c=np.conj(v)
    site_set=set(sample.sites)
    a=np.zeros((N,N,N,N))
    b=np.zeros((N,N,N,N))
    c=np.zeros((N,N,N,N))
    d=np.zeros((N,N,N,N))
    for i in range(N):
        for j in range(N):
            for n in range(N):
                for m in range(N):
                    coord_i=sample.sites[i]
                    coord_ix= tuple(a + b for a, b in zip(coord_i, (1,0)))
                    coord_j=sample.sites[i]
                    coord_jx= tuple(a + b for a, b in zip(coord_j, (1,0)))
                    if (coord_ix in site_set) and (coord_jx in site_set):
                        i_x= sample.sites.index(coord_ix)
                        j_x= sample.sites.index(coord_jx)

                        a[i,j,n,m]=u[j,n]*u_c[i_x,n]*u[i,m]*u_c[j_x,m] - u_c[i_x,n]*v[j_x,n]*u[i,m]*v_c[j,m] - u[j_x,n]*u_c[i_x,n]*u[i,m]*u_c[j,m]\
                                   +u_c[i_x,n]*v[j,n]*u[i,m]*v_c[j_x,m] - u[j,n]*u_c[i,n]*u[i_x,m]*u_c[j_x,m]+ u_c[i,n]*v[j_x,n]*u[i_x,m]*v_c[j,m]\
                                   + u[j_x,n]*u_c[i,n]*u[i_x,m]*u_c[j,m] - u_c[i,n]*v[j,n]*u[i_x,m]*v_c[j_x,m]

                        b[i,j,n,m]=u[j,n]*u_c[i_x,n]*v_c[i,m]*v[j_x,m] + u_c[i_x,n]*v[j_x,n]*v_c[i,m]*u[j,m] - u[j_x,n]*u_c[i_x,n]*v_c[i,m]*v[j,m]\
                                   -u_c[i_x,n]*v[j,n]*v_c[i,m]*u[j_x,m] - u[j,n]*u_c[i,n]*v_c[i_x,m]*v[j_x,m]- u_c[i,n]*v[j_x,n]*v_c[i_x,m]*u[j,m]\
                                   + u[j_x,n]*u_c[i,n]*v_c[i_x,m]*v[j,m] + u_c[i,n]*v[j,n]*v_c[i_x,m]*u[j_x,m]

                        c[i, j, n, m] =v_c[j,n]*v[i_x,n]*u[i,m]*u_c[j_x,m] + v[i_x,n]*u_c[j_x,n]*u[i,m]*v_c[j,m] - v_c[j_x,n]*v[i_x,n]*u[i,m]*u_c[j,m]\
                                   -v[i_x,n]*u_c[j,n]*u[i,m]*v_c[j_x,m] - v_c[j,n]*v[i,n]*u[i_x,m]*u_c[j_x,m]- v[i,n]*u_c[j_x,n]*u[i_x,m]*v_c[j,m]\
                                   + v_c[j_x,n]*v[i,n]*u[i_x,m]*u_c[j,m] + v[i,n]*u_c[j,n]*u[i_x,m]*v_c[j_x,m]

                        d[i, j, n, m] = v_c[j,n]*v[i_x,n]*v_c[i,m]*v[j_x,m] - v[i_x,n]*u_c[j_x,n]*v_c[i,m]*u[j,m] - v_c[j_x,n]*v[i_x,n]*v_c[i,m]*v[j,m]\
                                   +v[i_x,n]*u_c[j,n]*v_c[i,m]*u[j_x,m] - v_c[j,n]*v[i,n]*v_c[i_x,m]*v[j_x,m]+ v[i,n]*u_c[j_x,n]*v_c[i_x,m]*u[j,m]\
                                   + v_c[j_x,n]*v[i,n]*v_c[i_x,m]*v[j,m] - v[i,n]*u_c[j,n]*v_c[i_x,m]*u[j_x,m]

                        Lambda_part[i,j,n,m]=hopping**2/N*np.exp(1j*q_y*(coord_i[1]-coord_j[1]))*((a[i,j,n,m]+d[i,j,n,m])*(F(spectra[N+n],mu,T)-F(spectra[N+m],mu,T))/(spectra[N+n]-spectra[N+m])
                                                                    +(b[i,j,n,m]+c[i,j,n,m])*(F(spectra[N+n],mu,T)+F(spectra[N+m],mu,T))/(spectra[N+n]+spectra[N+m]))
    Lambda=np.sum(Lambda_part, axis=(1,2,3))
    return Lambda
    
def local_stiffness(sample, H_2p,hopping, mu,T, q_y):
    
    K=local_kinetic_energy(sample, H_2p, hopping, mu, T)
    Lambda=twopoint_correlator(sample, H_2p,hopping, mu,T, q_y)
    return K-Lambda


def construct_hamiltonian(N, H, Delta):
    H_Delta = diags([Delta], [0], shape=(N, N)).toarray()
    H_bdg = np.block([[H, H_Delta], [H_Delta, -H]])
    return H_bdg


def charge_density(sample, H_2p,mu,T):
    N=len(sample.sites)
    spectra, vectors = eigh(H_2p)
    energies=F(spectra[N:],mu,T)
    #print(energies)
    v=vectors[N:,N:]
    u=vectors[:N,N:]
    n=2*np.einsum(u,[0,1],np.conj(u), [0,1],energies,[1],[0])+2*np.einsum(v,[0,1],np.conj(v), [0,1],np.ones(N)-energies,[1],[0])

    return n


def BdG_cycle(sample, V,T,Delta_ini=0, initial=True):

    H_1p=sample.hamiltonian
    N=len(sample.sites)
    print("N",N)
    if initial==False:
        Delta=0.1*np.ones(N)+0.1*np.random.rand(N)
    else:
        Delta = Delta_ini
    step=0

    while True:
        H=construct_hamiltonian(N,H_1p, Delta)
        spectra, vectors = eigh(H)
        vectors_up=0.5 * V * vectors[N:,:]
        Delta_next= np.einsum(vectors_up, [0,1], vectors[:N,:], [0,1], np.tanh(spectra/ T),[1],[0])
        error=np.max(np.abs((Delta-Delta_next)))
        Delta=Delta_next
        print("step", step, "error", error, "Delta_max", np.max(np.abs(Delta)))
        step += 1
        if error<10**(-6):
            break

    return Delta, H


#function to calculate different variants#
def V_T_cycle(t,size):
    mode = "triangle"
    # N=27*27
    # dis_list=[]
    # for i in range(N):
    #     x = random.random_sample()
    #     if x >= 0.9:
    #         dis_list.append(i)

    sample_lat = Lattice(t, mode, size)
    sample_lat.create_hamiltonian()
    sample_frac_1 = Lattice(t, mode, size, fractal_iter=1)
    sample_frac_1.create_hamiltonian()
    sample_frac_2 = Lattice(t, mode, size, fractal_iter=2)
    sample_frac_2.create_hamiltonian()
    sample_frac_3= Lattice(t, mode, size, fractal_iter=4)
    sample_frac_3.create_hamiltonian()
    sample_dis_1 = Lattice(t, mode, size, alpha=0.1)
    sample_dis_1.create_hamiltonian()
    sample_dis_2 = Lattice(t, mode, size, alpha=0.2)
    sample_dis_2.create_hamiltonian()
    sample_dis_3 = Lattice(t, mode, size, alpha=0.3)
    sample_dis_3.create_hamiltonian()
    for j in range(2):

        V=1.5+1.5*j
        #print("params T=", T, " V=", V)
        N_iter=50
        Delta_lattice=[]
        Delta_fractal_1=[]
        Delta_fractal_2 = []
        Delta_fractal_3 = []
        Delta_disorder_1=[]
        Delta_disorder_2 = []
        Delta_disorder_3 = []
        for i in range(N_iter):

            T=0.01+0.01*i
            print("V=",V,"T=", T)
            print("BdG lattice triangle 33")

            if i==0:
                Delta, H = BdG_cycle(sample_lat, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_lat, V, T, Delta_ini=Delta_lattice_ini)
            Delta_lattice.append(Delta)
            Delta_lattice_ini=np.copy(Delta)
            #
            print("BdG fractal 1iter")

            if i==0:
                Delta, H = BdG_cycle(sample_frac_1, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_frac_1, V, T, Delta_ini=Delta_fractal1_ini)
            Delta_fractal_1.append(Delta)
            Delta_fractal1_ini = np.copy(Delta)

            print("BdG fractal 2iter")

            if i==0:
                Delta, H = BdG_cycle(sample_frac_2, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_frac_2, V, T, Delta_ini=Delta_fractal2_ini)
            Delta_fractal_2.append(Delta)
            Delta_fractal2_ini = np.copy(Delta)

            print("BdG fractal 4iter")

            if i==0:
                Delta, H = BdG_cycle(sample_frac_3, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_frac_3, V, T, Delta_ini=Delta_fractal3_ini)
            Delta_fractal_3.append(Delta)
            Delta_fractal3_ini = np.copy(Delta)

            print("BdG lattice 10% disorder")
            if i==0:
                Delta, H = BdG_cycle(sample_dis_1, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_dis_1, V, T, Delta_ini=Delta_disorder1_ini)
            Delta_disorder_1.append(Delta)
            Delta_disorder1_ini = np.copy(Delta)

            print("BdG lattice 20% disorder")

            if i==0:
                Delta, H = BdG_cycle(sample_dis_2, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_dis_2, V, T, Delta_ini=Delta_disorder2_ini)
            Delta_disorder_2.append(Delta)
            Delta_disorder2_ini = np.copy(Delta)
            print("BdG lattice 30% disorder")

            if i==0:
                Delta, H = BdG_cycle(sample_dis_3, V, T, initial=False)
            else:
                Delta, H = BdG_cycle(sample_dis_3, V, T, Delta_ini=Delta_disorder3_ini)
            Delta_disorder_3.append(Delta)
            Delta_disorder3_ini = np.copy(Delta)

        Delta_lattice = np.array(Delta_lattice)
        Delta_fractal_1=np.array(Delta_fractal_1)
        Delta_fractal_2=np.array(Delta_fractal_2)
        Delta_fractal_3=np.array(Delta_fractal_3)
        Delta_disorder_1=np.array(Delta_disorder_1)
        Delta_disorder_2=np.array(Delta_disorder_2)
        Delta_disorder_3=np.array(Delta_disorder_3)
        np.save("V="+str(V)+"_triangle.npy", Delta_lattice)
        np.save("V=" + str(V) + "_gasket_1iter.npy", Delta_fractal_1)
        np.save("V=" + str(V) + "_gasket_2iter.npy", Delta_fractal_2)
        np.save("V=" + str(V) + "_gasket_4iter.npy", Delta_fractal_3)
        np.save("V=" + str(V) + "_triangle_disorder_1.npy", Delta_disorder_1)
        np.save("V=" + str(V) + "_triangle_disorder_2.npy", Delta_disorder_2)
        np.save("V=" + str(V) + "_triangle_disorder_3.npy", Delta_disorder_3)

    return

def plot_figures():

    T=np.linspace(0.01, 0.5, num=50)
    #print(T.shape)
    type=["triangle", "triangle_disorder_1", "triangle_disorder_2", "triangle_disorder_3", "gasket_1iter", "gasket_2iter", "gasket_4iter"]
    for str_type in type:
        file_name="V=1.5_"+str_type+".npy"
        Delta_T_array=np.load(file_name)
        #print(Delta_T_array.shape)
        N=Delta_T_array.shape[0]
        Delta_max=np.zeros(N)
        for i in range(N):
            Delta_max[i]=np.max(np.abs(Delta_T_array[i,:]))
        if str_type=="triangle":
            legend="triangle L=33"
        if str_type == "triangle_disorder_1":
            legend = "10% holes"
        if str_type == "triangle_disorder_2":
            legend = "20% holes"
        if str_type == "triangle_disorder_3":
            legend = "30% holes"
        if str_type == "gasket_1iter":
            legend = "gasket 1 iteration"
        if str_type == "gasket_2iter":
            legend = "gasket 2 iteration"
        if str_type == "gasket_4iter":
            legend = "gasket 4 iteration"
        plt.plot(T, Delta_max, label=legend)
        plt.xlabel("T")
        plt.ylabel(r'$\Delta_{max}$')
        plt.title("V=1.5")
    plt.legend()
    plt.savefig("V=1.5_gasket_disorder.png")
    plt.close()

def show_graph(pos):
    plt.scatter(pos[:,0], pos[:,1], s=4)
    plt.savefig("graph.png")
    plt.close()

def main():

    #plot_figures()
    # t=1
    # size=33
    # V_T_cycle(t,size)

    #
    # mode="triangle"
    # t=1
    # size=13
    # V=1.5
    # T=0.2
    # sample=Lattice(t,mode,size,fractal_iter=0)
    # sample.create_hamiltonian()
    # H=sample.hamiltonian
    # spectra_1, vectors = eigh(H)
    #print(spectra)

    #BdG_cycle(sample, V, T, initial=False)
    #
    # sample=Lattice(t,mode,size)
    # BdG_cycle(sample, V, T, initial=False)

    mode="square"
    t=1
    size=27
    T=1/20
    V=0
    mu=10**(-2)
    sample = Lattice(t, mode, size, fractal_iter=0, pbc=True)
    sample.create_hamiltonian()
    Delta, H = BdG_cycle(sample, V, T)
    n=charge_density(sample, H, mu, T)
    print("charge density", np.sum(n)/len(sample.sites))
    K=local_kinetic_energy(sample,H,t,mu,T)
    print("kinetic energy", np.sum(K)/len(sample.sites))


main()