import numpy as np
import BdG_on_graph as bdg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
import test_functions as test
import networkx as nx

def main():

    t=1
    T_array=np.array([0.005, 0.05])
    V_array=np.array([1])
    mu_array=np.array([2])  
    fractal_iter=0
    alpha=0.3
    
    size=33
    mode="triangle"
    lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=False, noise=True)
    bdg.calculate_diagram(lattice_sample,V_array, mu_array, T_array)


         
main()
