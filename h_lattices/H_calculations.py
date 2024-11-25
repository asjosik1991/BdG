
import numpy as np
import Hyperbolic_BdG as hbdg
import matplotlib.pyplot as plt

def main():

    p=8
    q=3
    l=8
    t=1
    V=1
    T=0.01
    mu=0
    
    #hypersample=hbdg.HyperLattice(p,q,l,t) #Hyperbolic lattice

    #hypersample=hbdg.HyperLattice(p,q,l,t, loadfile="adjs for h_lattices/8_3_256sites.mtx") #Hyperbolic lattice
    
    #plot phase diagram
    # V_array=[2]
    # T_array=np.linspace(0.01, 0.3, num=30)
    # mu_array=np.linspace(-3, 3, num=30)

    # hbdg.calculate_hyperdiagram(hypersample,V_array,mu_array,T_array, uniform=True)
    # hdiagram=hbdg.load_hyperdiagram(hypersample)
    # hbdg.plot_diagram(hypersample, hdiagram)
    
    
    "test tree graph"
    
    hypersample=hbdg.Tree_graph(q,l,t) #Tree graph
    BdGhypersample=hbdg.HyperBdG(hypersample,V,T,mu)
    BdGhypersample.BdG_cycle()
    BdGhypersample.nx_Delta_plot()

    #BdGhypersample.field_plot(np.round(BdGhypersample.Delta,4))
    BdGhypersample.plot_radial_Delta()
   
    "test tree graph"
    
    hypersample=hbdg.centered_HL(l)
    BdGhypersample=hbdg.HyperBdG(hypersample,V,T,mu)
    BdGhypersample.BdG_cycle()
    BdGhypersample.nx_Delta_plot()
    BdGhypersample.plot_radial_Delta()





         
main()
