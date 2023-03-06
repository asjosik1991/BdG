import numpy as np
import BdG_on_graph as bdg

def main():

    mode="square"
    t=1
    size=1001
    T=0.01
    V=1.0
    mu=0.5
    fractal_iter=0
    alpha=0.0
    
    bdg.uniform_2D_BdG(size,mu,V,T)
    
    # T_array=np.array([T])
    # V_array=np.array([V])
    # mu_array=np.array([mu])
    # lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=True, noise=True)
    # bdg.calculate_diagram(lattice_sample, V_array, mu_array, T_array)
    # diagram=bdg.load_diagram(lattice_sample)
    # bdg.plot_diagram(diagram, 'mu',show=True)
    
    # BdG_sample=bdg.BdG(lattice_sample, V, T, mu)
    # bdg.BdG_sample.BdG_cycle()
    # print(lattice_sample.hamiltonian)

    # rho=bdg.BdG_sample.local_kinetic_energy()
    # print("rho", np.real(rho), "rho_av", np.mean(np.real(rho)))
    # bdg.BdG_sample.field_plot(np.real(np.round(rho,4)))


         
main()