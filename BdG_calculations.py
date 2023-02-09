import numpy as np
import BdG_on_graph as bdg

def main():

    mode="triangle"
    t=1
    size=17
    T=0.1
    V=2.0
    mu=0.0
    fractal_iter=1
    alpha=0.0
    T_array=np.array([0.1])
    V_array=np.array([2.0])
    mu_array=np.linspace(-4,4,40)
    lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=True, noise=True)
    bdg.calculate_diagram(lattice_sample, V_array, mu_array, T_array)
    diagram=bdg.load_diagram(lattice_sample)
    bdg.plot_diagram(diagram, 'mu')
    
    # BdG_sample=bdg.BdG(lattice_sample, V, T, mu)
    # bdg.BdG_sample.BdG_cycle()
    # print(lattice_sample.hamiltonian)

    # rho=bdg.BdG_sample.local_kinetic_energy()
    # print("rho", np.real(rho), "rho_av", np.mean(np.real(rho)))
    # bdg.BdG_sample.field_plot(np.real(np.round(rho,4)))


         
main()