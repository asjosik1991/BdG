import numpy as np
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import bisect
import matplotlib.cm as cmap

def build_radial_matrix_hyperbolic(m, rgrid):
    """
    Construct the finite-difference matrix A for the radial operator:
        A = -Delta_h (restricted to radial part) for a given m,
    where
        Delta_h = 1/sinh(r) * d/dr [ sinh(r) d/dr ] + 1/sinh^2(r)*d^2/dtheta^2.
    After separating out e^{i m theta}, we get an effective radial operator:
        -Delta_h(R_m) = -[ 1/sinh(r) d/dr( sinh(r) dR_m/dr ) - m^2 / sinh^2(r) R_m ].
    We'll build a finite-difference approximation of that operator.
    
    Parameters
    ----------
    m : int
        Angular quantum number (0, 1, 2, ...).
    rgrid : 1D ndarray
        The radial grid points (excluding the outer boundary r=R).
    
    Returns
    -------
    A : 2D ndarray
        Symmetric matrix whose eigen-decomposition approximates
        -Delta_h R_m = lambda * R_m.
    """
    dr = rgrid[1] - rgrid[0]
    npts = len(rgrid)
    A = np.zeros((npts, npts), dtype=np.float64)
    
    # Helper for repeated factor:
    #    c(r) = 1 / sinh(r),   d(r) = 1 / sinh^2(r).
    def d(r): return 4.0*np.exp(2*r)
    
    for i in range(npts):
        r_i = rgrid[i]
        
        # Diagonal part (second derivative + m^2/r^2)
        diag_val = 2.0 / (dr * dr) + d(r_i)*(m*m)+0.25
        A[i, i] = diag_val
        
        # Off-diagonals (second derivative)
        if i > 0:
            A[i, i-1] = -1.0 / (dr * dr)
        if i < npts - 1:
            A[i, i+1] = -1.0 / (dr * dr)
    
    return A


def solve_radial_equation_hyperbolic(m, rmin, rmax, nr):
    """
    Solve the hyperbolic radial eigenvalue problem for:
       -Delta_h( R_m(r) e^{i m theta} ) = lambda * R_m(r) e^{i m theta},
    with ds^2 = dr^2 + sinh^2(r) dtheta^2, on [rmin, rmax].
    Dirichlet boundary: R_m(rmax)=0. 
    For demonstration, we also treat rmin>0 with R_m(rmin)=0 (good for m>0).
    Returns the discrete eigenvalues and eigenfunctions.
    """
    # Build radial grid, skipping exactly r=0 and r=rmax
    rgrid = np.linspace(rmin, rmax, nr+1, endpoint=False)[1:]  # interior points only
    
    # Build the matrix
    A = build_radial_matrix_hyperbolic(m, rgrid)
    
    # Diagonalize A v = lambda v
    eigvals, eigvecs = eig(A)
    
    # Sort
    idx_sort = np.argsort(eigvals)
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]
    
    return rgrid, eigvals, eigvecs


def plot_density_of_states(results, Emax=None, nbins=50):
    """
    Plot a rough density of states (DOS) from the eigenvalues in 'results'.
    results[m] = { 'rgrid':..., 'eigenvalues':..., 'eigenvectors':... }
    """
    all_eigvals = []
    for m in results:
        if m>0:
            all_eigvals.append(results[m]['eigenvalues'])
            all_eigvals.append(results[m]['eigenvalues'])
    all_eigvals = np.concatenate(all_eigvals)
    
    if Emax is not None:
        all_eigvals = all_eigvals[all_eigvals < Emax]
    
    if len(all_eigvals) == 0:
        print("No eigenvalues below Emax.")
        return
    
    # Basic histogram
    if Emax is None:
        Emax = max(all_eigvals)
    hist, bins = np.histogram(all_eigvals, bins=nbins, range=(-100, Emax))
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    dos = hist / bin_width
    
    plt.figure(figsize=(6,4))
    plt.plot(bin_centers, dos, drawstyle='steps-mid')
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel("Density of States")
    plt.title("Hyperbolic Disk")
    plt.tight_layout()
    plt.show()


class BCS_hyper:
    
    def __init__(self,r_max, r_min, m_array, nr, V,T, e_min, e_max,n_m=400):
        self.r_max=r_max
        self.nr=nr
        self.m_array=m_array
        self.m_max=np.max(m_array)
        self.n_m=n_m
        self.V=V*np.pi*np.sqrt(2)
        self.T=T
        self.e_min=e_min
        self.e_max=e_max
        self.mu=-5
        self.Delta=[]
        self.rgrid = np.linspace(r_min, r_max, nr) 
        self.e_fermi=0#(self.e_min+self.e_max)/2
        self.N=len(self.rgrid)
        self.dm=self.m_array[1] - self.m_array[0]
        self.dr=self.rgrid[1] - self.rgrid[0]
    
    #Fermi function
    def F(self,E):
        return 1/(np.exp(E/self.T)+1)    
    
    def effective_H(self,m):
        
        
        #dr = self.rgrid[1] - self.rgrid[0]
        npts = len(self.rgrid)
        H = np.zeros((npts, npts), dtype=np.float64)
        
        # Helper for repeated factor:
        #    c(r) = 1 / sinh(r),   d(r) = 1 / sinh^2(r).
        def d(r): return 4.0*np.exp(2*r)
        
        for i in range(npts):
            r_i = self.rgrid[i]
            
            # Diagonal part (second derivative + m^2/r^2)
            diag_val = -2.0 / (self.dr * self.dr) - d(r_i)*(m*m)-0.25
            H[i, i] = diag_val
            
            # Off-diagonals (second derivative)
            if i > 0:
                H[i, i-1] = 1.0 / (self.dr * self.dr)
            if i < npts - 1:
                H[i, i+1] = 1.0 / (self.dr * self.dr)
        
        return H
    
    def effective_BdG(self, m,Delta):
        H=self.effective_H(m)
        Delta=np.diag(Delta)
        BdG_H = np.block([[H- self.mu*np.eye(self.N), Delta], [Delta, -H+self.mu*np.eye(self.N)]])
        return BdG_H
    
    def choose_inds(self, energies):
        
        left_index = bisect.bisect_left(energies, self.e_min)    
        right_index = bisect.bisect_right(energies, self.e_max) - 1
        return left_index, right_index
    
    # def normalize_vectors(self, vectors):
    #     A = 1/np.sqrt(self.rgrid[1] - self.rgrid[0])
    #     exp_vector=A*np.exp(-0.5*self.rgrid)

    #     vectors_up=np.einsum(exp_vector,[0],np.copy(vectors[self.N:,:]),[0,1],[0,1])
    #     vectors_down=np.einsum(exp_vector,[0],np.copy(vectors[:self.N,:]),[0,1],[0,1])
        
    #     return vectors_up, vectors_down
        
    
    def gap_integral(self,Delta):
        
        npts = len(self.rgrid)
        gap=np.zeros(npts)
        m_max_new=0
        for m in self.m_array:
            #print("m=",m)
            BdG_H=self.effective_BdG(m,Delta)
            spectra, vectors = eigh(BdG_H,subset_by_value=[self.e_min, self.e_max])
            N_spec=len(spectra)
            #if N_spec>0:
            #    print(m, N_spec, np.min(spectra), np.max(spectra))
            if N_spec==0:
                print("max_m=", m)
                m_max_new=m
                break
            F_weight=np.ones(N_spec)-2*self.F(spectra)
            vectors_up=vectors[self.N:,:]
            vectors_down=vectors[:self.N,:]
            #vectors_up, vectors_down=self.normalize_vectors(vectors)
                            
            if m==0:
                #print("number of eigenstates for m=0", N_spec)
                vectors_up=self.dm*self.V*vectors_up
            else:
                vectors_up=self.dm*2*self.V*vectors_up
                    
            gap+=np.einsum(vectors_up, [0,1], vectors_down, [0,1], F_weight,[1],[0])
        
        exp_array=1/self.dr*np.exp(self.rgrid)
        gap=exp_array*gap
        
        if m_max_new<0.5*self.m_max:
            self.m_max=2*m_max_new
            self.m_array=np.linspace(0,self.m_max,self.n_m)
            self.dm=self.m_array[1] - self.m_array[0]

        return gap
    
        
    def BdG_cycle(self):
        
        step=0
        h=0.9
        self.Delta=0.1*np.ones(self.N)+0.001*np.random.rand(self.N)             
        while True:
            Delta_next=(1-h)*self.Delta + h*self.gap_integral(self.Delta)
            error=np.max(np.abs((self.Delta-Delta_next)))
            self.Delta=Delta_next
            #self.plot_Delta()
            print("step", step, "error", error, "Delta_max", np.max(np.abs(self.Delta)))
            step += 1
            if error<10**(-6):
                break
        
        #self.Delta=Delta
        return self.Delta
    
    def plot_Delta(self):
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel(r'$\Delta$',fontsize=20)
        plt.xlabel(r'$r$',fontsize=20)
        plt.plot(self.rgrid, self.Delta)

        plt.title("Hyper disc", fontsize=20)
        #plt.title("Bethe lattice DoS", fontsize=20)

        plt.show()
        plt.close()

def plot_boundary_state(): #plot
    
    "Calculution"
    r_min = 0  # small radius near zero (to avoid singularity)
    r_max = 10     # "radius" of the hyperbolic disk
    nr   = 200    # number of radial steps
    m_array=np.linspace(0,10,100)
    rgrid = np.linspace(r_min, r_max, nr) 

    
    T=0.5
    V=1
    e_min=0
    e_max=7
    disc=BCS_hyper(r_max, r_min, m_array, nr, V,T, e_min, e_max)
    disc.BdG_cycle()
    
    "Plotting"
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(3.4,2.5),dpi=1000,layout='constrained')
    ax.locator_params(nbins=7)
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    plt.xlabel(r'x',fontsize=8,labelpad=1)
    plt.title(r"Boundary state", fontsize=12,y=1.02)
    plt.plot(rgrid,disc.Delta, linewidth=1.1,color='royalblue')
    
    
    plt.show()
    # filename="hyperbolic_boundary_state.pdf"
    # plt.savefig(filename)
    # plt.close()
    
    return

def main():
    
    # r_min = 0  # small radius near zero (to avoid singularity)
    # r_max = 2     # "radius" of the hyperbolic disk
    # nr   = 200   # number of radial steps
    # #nm  = 15000     # solve for m = 0..5    
    # m_array=np.linspace(0,5,100)
    # ## Dictionary to hold results
    
    # results = {}
    # for m in m_array:
    #     rgrid, eigvals, eigvecs = solve_radial_equation_hyperbolic(m, r_min, r_max, nr)
    #     results[m] = {
    #         'rgrid': rgrid,
    #         'eigenvalues': eigvals,
    #         'eigenvectors': eigvecs
    #     }
        
    #     # Print a few lowest eigenvalues for each m
    #     print(f"m = {m}, first 5 eigenvalues ~ {eigvals[:5]}")
    # plot_density_of_states(results, Emax=10, nbins=50)
    
    # m_plot = m_array[9]
    # rgrid = results[m_plot]['rgrid']
    # eigvals = results[m_plot]['eigenvalues']
    # eigvecs = results[m_plot]['eigenvectors']
    
    # plt.figure(figsize=(7,5))
    # for i in range(10):
    #     plt.plot(rgrid, eigvecs[:, i],label=f"Mode {i}, λ={eigvals[i]:.3f}")
    #     print(m_plot, eigvals[i])
    # plt.xlabel("r")
    # plt.ylabel(r"$R_{m}(r)$")
    # plt.title(f"Radial eigenfunctions for m={m_plot}")
    # #plt.legend()
    # plt.show()
    
    # # T=0.1
    # # V=35
    # # e_min=0
    # # e_max=5
    # # disc=BCS_hyper(r_max, r_min, m_array, nr, V,T, e_min, e_max)
    # # disc.BdG_cycle()
    # # disc.plot_Delta()
    plot_boundary_state()

if __name__ == "__main__":
    main()
