import numpy as np
import BdG_on_graph as bdg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
import test_functions as test
import networkx as nx

def main():

    "Fractals"
    t=1
    T=0.005
    V=1
    mu=2
    fractal_iter=4
    alpha=0 
    size=33
    mode="regular_triangle"
    
    lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=False, noise=True)
    BdG_sample=bdg.BdG(lattice_sample, V, T, mu)
    BdG_sample.BdG_cycle()
    BdG_sample.field_plot(BdG_sample.Delta, fieldname='Delta', title='$\Delta$', edges=True, contrast=True)
    rho=BdG_sample.local_stiffness(0.0001)
    BdG_sample.field_plot(rho, fieldname='rho', title='$D_s/\pi$',edges=True, contrast=True)





         
main()
