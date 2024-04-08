import numpy as np
import Hyperbolic_BdG as hbdg
import matplotlib.pyplot as plt

def main():

    p=8
    q=3
    l=3
    t=1
    V=2
    T=0.02
    mu=0
    
    #hypersample=hbdg.HyperLattice(p,q,l,t) #Hyperbolic lattice

    hypersample=hbdg.HyperLattice(p,q,l,t, loadfile="adjs for h_lattices/8_3_256sites.mtx") #Hyperbolic lattice
    #hypersample=hbdg.Tree_graph(q,l,t) #Tree graph
    
    #one BdG cycle and plotting the resu;t
    BdGhypersample=hbdg.HyperBdG(hypersample,V,T,mu, uniform=True)
    BdGhypersample.plot_lattice_spectrum()
    #BdGhypersample.BdG_cycle()
    # #print(BdGhypersample.Delta)
    # BdGhypersample.field_plot(np.round(BdGhypersample.Delta,4))
    
    #plot phase diagram
    V_array=[2]
    T_array=np.linspace(0.01, 0.3, num=30)
    mu_array=np.linspace(-3, 3, num=30)

    hbdg.calculate_hyperdiagram(hypersample,V_array,mu_array,T_array, uniform=True)
    hdiagram=hbdg.load_hyperdiagram(hypersample)
    hbdg.plot_diagram(hypersample, hdiagram)
    





         
main()
