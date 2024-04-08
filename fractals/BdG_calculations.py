import numpy as np
import BdG_on_graph as bdg
import matplotlib.pyplot as plt

def main():

    "Fractals"
    t=1
    T=0.005
    V=1
    mu=1.15
    fractal_iter=4
    alpha=0 
    size=33
    mode="regular_triangle"
    
    # t=1
    # T=0.005
    # V=1
    # mu=-0.5
    # fractal_iter=0
    # alpha=0 
    # size=24
    # mode="square"
    
    lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=False, noise=False)
    BdG_sample=bdg.BdG(lattice_sample, V, T, mu, HF=True)
    BdG_sample.BdG_cycle()
    BdG_sample.field_plot(np.round(BdG_sample.Delta,3), fieldname='Delta', title='$\Delta$', edges=True, contrast=True)
    rho=BdG_sample.local_stiffness(0.0001)
    BdG_sample.field_plot(np.round(rho,3), fieldname='rho', title='$D_s/\pi$',edges=True, contrast=False)





         
main()
