import numpy as np
import Hyperbolic_BdG as hbdg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
import test_functions as test
import networkx as nx

def main():

    p=8
    q=3
    l=3
    t=1
    # V=5
    # T=0.02
    # mu=0
    hypersample=hbdg.HyperLattice(p,q,l,t)
    # #print(hypersample.hamiltonian)
    # #print(hypersample.sites)
    # BdGhypersample=HyperBdG(hypersample,V,T,mu)
    # BdGhypersample.BdG_cycle()
    # #print(BdGhypersample.Delta)
    # BdGhypersample.field_plot(np.round(BdGhypersample.Delta,4))
    
    V_array=[2]
    T_array=np.linspace(0.01, 0.3, num=10)
    mu_array=np.linspace(-3, 3, num=10)

    hbdg.calculate_hyperdiagram(hypersample,V_array,mu_array,T_array)
    hdiagram=hbdg.load_hyperdiagram(hypersample)
    hbdg.plot_diagram(hypersample, hdiagram)





         
main()
