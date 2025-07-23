import numpy as np
import hyper_aux.aux_functions as haux
import Hyperbolic_BdG as hbdg
import effective_models.Cayley_tree as tree

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

from hypertiling import HyperbolicGraph, GraphKernels, HyperbolicTiling
from hypertiling.graphics.plot import plot_tiling
from hypertiling.kernel.GRG_util import plot_graph
import networkx as nx


import matplotlib.cm as cmap
palette = ["#81b29a", "#f2cc8f", "#e07a5f"]


def plot_DoS_phasediag():     #Fig.1
    
    "Calculations"
    energy_range=np.linspace(-4.0, 4.0, num=150)
    dos_array=[]
    for s in energy_range:
        dos_array.append(haux.hyper_dos83(s))
    
    V_array=[1]
    mu_array=energy_range
    T_array=np.linspace(0.001, 0.025, num=200)
    haux.calculate_exactdiagram(haux.hyper_dos83, V_array, mu_array, T_array)
    hyper=haux.load_exactdiagram()
    if len(hyper['V'])==1:
        V=hyper['V'][0]
        x=hyper['mu']
        y=hyper['T']
        Deltas=hyper['Deltas']
        print(x)
        print(y)
        print(Deltas)
        Delta=np.zeros((len(y),len(x)))      

        for i in range(len(y)):
            for j in range(len(x)):
                Delta[i,j]=Deltas[(V,y[i],x[j])]
    field=Delta
    
    "Plots"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    
    ax1.plot(energy_range, dos_array,linewidth=1.9,color='royalblue')
    ax1.locator_params(nbins=5)
    ax1.set_xlabel(r'$s$', fontsize=22)
    ax1.set_ylabel(r'$\rho(s)$', fontsize=22)
    ax1.tick_params(labelsize=18)
    ax1.set_box_aspect(1)
    ax1.set_title("\{8,3\} lattice, DoS", fontsize=26, y=1.02)
    
    pcm=ax2.imshow(field, vmin=field.min(), vmax=field.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
    ax2.locator_params(nbins=5)
    ax2.set_xlabel(r'$\mu$', fontsize=22)
    ax2.set_ylabel(r'$T$', fontsize=22)
    ax2.tick_params(labelsize=18)
    ax2.set_box_aspect(1)
    ax2.set_title("\{8,3\} lattice, Phase diagram", fontsize=26, y=1.02)
    #ax2.set_title('b)', fontfamily='serif', loc='left', fontsize=26, y=1.15,x=-0.2)
    cbar=fig.colorbar(pcm, ax=ax2, shrink=0.6, location='top',pad=-0.05)
    cbar.set_label(label=r'$\Delta$',fontsize=22, rotation=0,x=1.1, labelpad=-30)
    cbar.ax.tick_params(labelsize=18)    
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax1.text(-0.1,1.11,'a)', transform=ax1.transAxes, fontsize=28, fontstyle='oblique')
    ax2.text(-0.12,1.11,'b)', transform=ax2.transAxes, fontsize=28, fontstyle='oblique')


    #plt.show()

    filename="8_3_bulk_example.pdf"
    plt.savefig(filename)
    plt.close()

def plot_hyper_lattices():    #Fig.2
    
    "Calculations"
    q=2
    p=8
    l1=8
    l2=11
    t=1
    V=1
    mu=0
    T=0.01
    
    hypersample1=hbdg.centered_HL(l1) 
    h_sample1=hbdg.HyperBdG(hypersample1,V,T,mu)
    h_sample1.BdG_cycle()
    r_Delta1, r_sigma1=h_sample1.get_radial_Delta()
    
    CT1=tree.Caylee_tree(q, l1, V, T, mu)
    CT1.BdG_cycle()
    r_CDelta1=CT1.Delta
    
    hypersample3=hbdg.centered_HL(l2) 
    h_sample2=hbdg.HyperBdG(hypersample3,V,T,mu)
    h_sample2.BdG_cycle()
    r_Delta2, r_sigma2=h_sample2.get_radial_Delta()
    
    CT2=tree.Caylee_tree(q, l2, V, T, mu)
    CT2.BdG_cycle()
    r_CDelta2=CT2.Delta
    
    
    "Plots"
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=10 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    
    axs[0,0].plot(r_Delta1, label="{\{8,3\} lattice, $\Delta$", linewidth=1.9,color='royalblue')
    axs[0,0].plot(r_sigma1, label="{\{8,3\} lattice, $\sigma$", linewidth=1.9,color='coral')
    axs[0,0].plot(r_CDelta1, label="Cayley tree, $\Delta$", linewidth=1.9, color='slategray')

    axs[0,0].locator_params(nbins=5)
    axs[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[0,0].set_xlabel(r'Distance from the center', fontsize=22)
    axs[0,0].set_ylabel(r'$\Delta$', fontsize=22)
    axs[0,0].tick_params(labelsize=18)
    axs[0,0].set_box_aspect(1)
    axs[0,0].set_title("Radial $\Delta$, 8 shells", fontsize=26, y=1.02)
    axs[0,0].legend(fontsize=22)
    
    axs[1,0].plot(r_Delta2, label="{\{8,3\} lattice, $\Delta$", linewidth=1.9,color='royalblue')
    axs[1,0].plot(r_sigma2, label="{\{8,3\} lattice, $\sigma$", linewidth=1.9,color='coral')
    axs[1,0].plot(r_CDelta2, label="Cayley tree, $\Delta$", linewidth=1.9, color='slategray')

    axs[1,0].locator_params(nbins=5)
    axs[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1,0].set_xlabel(r'Distance from the center', fontsize=22)
    axs[1,0].set_ylabel(r'$\Delta$', fontsize=22)
    axs[1,0].tick_params(labelsize=18)
    axs[1,0].set_box_aspect(1)
    axs[1,0].set_title("Radial $\Delta$, 11 shells", fontsize=26, y=1.02)
    axs[1,0].legend(fontsize=22)

    
    colormap=plt.cm.plasma
    G1 = nx.from_numpy_array(h_sample1.lattice_H)
    nx.draw(G1,pos=nx.shell_layout(G1,nlist=h_sample1.lattice_sample.shell_list,rotate=0),ax=axs[0,1],node_color=h_sample1.Delta, node_size=700, node_shape='.',cmap=colormap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(h_sample2.Delta), vmax=max(h_sample2.Delta)))
    cbar1=fig.colorbar(sm,ax=axs[0,1],shrink=0.7)
    cbar1.ax.tick_params(labelsize=18)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar1.locator = tick_locator
    cbar1.set_label("$\Delta$", fontsize=22, rotation=0, labelpad=-25,y=1.1)
    cbar1.update_ticks()
    axs[0,1].set_title("\{8,3\} lattice, 8 shells", fontsize=26, y=1.02)

    G2 = nx.from_numpy_array(h_sample2.lattice_H)
    nx.draw(G2,pos=nx.shell_layout(G2,nlist=h_sample2.lattice_sample.shell_list,rotate=0),ax=axs[1,1],node_color=h_sample2.Delta, node_size=700, node_shape='.',cmap=colormap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(h_sample2.Delta), vmax=max(h_sample2.Delta)))
    cbar2=fig.colorbar(sm,ax=axs[1,1],shrink=0.7)
    cbar2.ax.tick_params(labelsize=18)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator
    cbar2.set_label("$\Delta$", fontsize=22, rotation=0, labelpad=-25,y=1.1)
    cbar2.update_ticks()
    axs[1,1].set_title("\{8,3\} lattice, 11 shells", fontsize=26, y=1.02)
    
    axs[0,0].text(-0.1,1.05,'a)', transform=axs[0,0].transAxes, fontsize=28, fontstyle='oblique')
    axs[0,1].text(-0.05,1.05,'b)', transform=axs[0,1].transAxes, fontsize=28, fontstyle='oblique')
    axs[1,0].text(-0.1,1.05,'c)', transform=axs[1,0].transAxes, fontsize=28, fontstyle='oblique')
    axs[1,1].text(-0.05,1.05,'d)', transform=axs[1,1].transAxes, fontsize=28, fontstyle='oblique')
    axs[0,0].set_box_aspect(1)
    axs[1,0].set_box_aspect(1)
    axs[0,1].set_box_aspect(1)
    axs[1,1].set_box_aspect(1)  

    #plt.show()
    
    filename="8_3_lattices_Delta.pdf"
    plt.savefig(filename)
    plt.close()

def plot_slice_phase_diagram():   #Fig.4
    "Calculation"
    q=2
    p=8
    l=8
    V=1
    mu=0
    Delta_bulk=[]
    Delta_edge=[]
    CDelta_bulk=[]
    CDelta_edge=[]
    
    hypersample=hbdg.centered_HL(l)
    T_array=np.linspace(0.01,0.21,60)
    for T in T_array:
        CT=tree.Caylee_tree(q, l, V, T, mu)
        CT.BdG_cycle()
        r_CDelta=CT.Delta
        
        h_sample=hbdg.HyperBdG(hypersample,V,T,mu)
        h_sample.BdG_cycle()
        r_Delta, r_sigma=h_sample.get_radial_Delta()
        
        Delta_bulk.append(r_Delta[0])
        Delta_edge.append(r_Delta[-1])
        CDelta_bulk.append(r_CDelta[0])
        CDelta_edge.append(r_CDelta[-1])
    
    
    "Plotting"
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(9,6.75))
    ax.locator_params(nbins=5)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(r'$\Delta$',fontsize=26)
    plt.xlabel(r'$T$',fontsize=26)
    plt.plot(T_array,Delta_edge, label='\{8,3\} lattice, edge', linewidth=1.9,color='royalblue')
    plt.plot(T_array,Delta_bulk, label="\{8,3\} lattice, center", linewidth=1.9,color='firebrick')
    plt.plot(T_array,CDelta_edge, label='Cayley tree, edge', linewidth=1.9,color='slategray')
    plt.plot(T_array,CDelta_bulk, label="Cayley tree, center", linewidth=1.9,color='orchid')
    plt.legend(fontsize=22)


    plt.title(r"Slice of the phase diagram", fontsize=32, y=1.02)

    #plt.show()
    filename="mu_slice_hyplat_tree.pdf"
    plt.savefig(filename)
    plt.close()

def plot_various_M():       #Fig.3

    "Calculation"
    q=2
    p=8
    T=0.01
    V=1
    mu=0
    #l_array=np.array([4,5,6,7,8,9,10,11])
    l_array=np.array([4,5,6])

    Delta_bulk1=[]
    Delta_edge1=[]
    CDelta_bulk1=[]
    CDelta_edge1=[]
    

    for l in l_array:
        CT=tree.Caylee_tree(q, l, V, T, mu)
        CT.BdG_cycle()
        r_CDelta=CT.Delta
        
        hypersample=hbdg.centered_HL(l)
        h_sample=hbdg.HyperBdG(hypersample,V,T,mu)
        h_sample.BdG_cycle()
        r_Delta, r_sigma=h_sample.get_radial_Delta()
        
        Delta_bulk1.append(r_Delta[0])
        Delta_edge1.append(r_Delta[-1])
        CDelta_bulk1.append(r_CDelta[0])
        CDelta_edge1.append(r_CDelta[-1])
    
    T=0.1
    Delta_bulk2=[]
    Delta_edge2=[]
    CDelta_bulk2=[]
    CDelta_edge2=[]
    
    for l in l_array:
        CT=tree.Caylee_tree(q, l, V, T, mu)
        CT.BdG_cycle()
        r_CDelta=CT.Delta
        
        hypersample=hbdg.centered_HL(l)
        h_sample=hbdg.HyperBdG(hypersample,V,T,mu)
        h_sample.BdG_cycle()
        r_Delta, r_sigma=h_sample.get_radial_Delta()
        
        Delta_bulk2.append(r_Delta[0])
        Delta_edge2.append(r_Delta[-1])
        CDelta_bulk2.append(r_CDelta[0])
        CDelta_edge2.append(r_CDelta[-1])
    
    
    "Plotting"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=10 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    plt.suptitle(r"Edge and center $\Delta$", fontsize=32, y=1.05)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=24)
    ax1.set_ylabel(r'$\Delta$',fontsize=26)
    ax1.set_xlabel(r'$M$',fontsize=26)
    ax1.plot(l_array,Delta_edge1, label='\{8,3\} lattice, edge', linewidth=1.9,color='royalblue')
    ax1.plot(l_array,Delta_bulk1, label="\{8,3\} lattice, center", linewidth=1.9,color='firebrick')
    ax1.plot(l_array,CDelta_edge1, label='Cayley tree, edge', linewidth=1.9,color='slategray')
    ax1.plot(l_array,CDelta_bulk1, label="Cayley tree, center", linewidth=1.9,color='orchid')
    ax1.legend(fontsize=22)
    ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=5)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=24)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'$M$',fontsize=26)
    ax2.plot(l_array,Delta_edge2, label='\{8,3\} lattice, edge', linewidth=1.9,color='royalblue')
    ax2.plot(l_array,Delta_bulk2, label="\{8,3\} lattice, center", linewidth=1.9,color='firebrick')
    ax2.plot(l_array,CDelta_edge2, label='Cayley tree, edge', linewidth=1.9,color='slategray')
    ax2.plot(l_array,CDelta_bulk2, label="Cayley tree, center", linewidth=1.9,color='orchid')
    #ax2.legend(fontsize=22)
    ax2.set_box_aspect(1)
    
    ax1.text(-0.1,1,'a)', transform=ax1.transAxes, fontsize=28, fontstyle='oblique')
    ax2.text(-0.1,1,'b)', transform=ax2.transAxes, fontsize=28, fontstyle='oblique')

    #plt.title(r"Slice of the phase diagram", fontsize=32, y=1.02)

    #plt.show()
    filename="various_M.pdf"
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    
    return 

def plot_profile_on_effective_Cayley_tree(): #Fig.5
    
    "Calculution"
    l=100
    T=0.1
    mu=0
    V=1
    CT=tree.effective_Caylee_HL(l,V,T,mu)
    CT.BdG_cycle()
    
    "Plotting"
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(9,6.75))
    ax.locator_params(nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(r'$\Delta$',fontsize=26)
    plt.xlabel(r'Distance from the center',fontsize=26)
    plt.title(r"Effective Cayley tree", fontsize=32, y=1.02)

    plt.plot(CT.Delta, linewidth=1.9,color='teal')
    
    
    #plt.show()
    filename="Delta_CTeff_V=1_M=100_T=0.1_q=2.pdf"
    plt.savefig(filename)
    plt.close()
    
    return

def plot_comparison_profiles(): #Fig. 6
    
    "Calculution"
    l=11
    T=0.1
    mu=0
    V=1
    q=2
    
    effCT=tree.effective_Caylee_HL(l,V,T,mu)
    effCT.BdG_cycle()
    
    CT=tree.Caylee_tree(q, l, V, T, mu)
    CT.BdG_cycle()
    
    hypersample=hbdg.centered_HL(l)
    h_sample=hbdg.HyperBdG(hypersample,V,T,mu)
    h_sample.BdG_cycle()
    r_Delta, r_sigma=h_sample.get_radial_Delta()
    
    "Plotting"
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(9,6.75))
    ax.locator_params(nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(r'$\Delta$',fontsize=26)
    plt.xlabel(r'Distance from the center',fontsize=26)
    plt.title(r"Radial $\Delta$", fontsize=32, y=1.02)
    
    plt.plot(r_Delta, label='\{8,3\} lattice', linewidth=1.9,color='royalblue')
    plt.plot(effCT.Delta, linewidth=1.9, label="effective Cayley tree", color='teal')
    plt.plot(CT.Delta, linewidth=1.9, label="Cayley tree", color='slategray')
    plt.legend(fontsize=22)
    
    #plt.show()
    filename="Radial_Deltas.pdf"
    plt.savefig(filename)
    plt.close()
    
    return

def compare_effective_trees():       #Fig.3

    "Calculation"
    q=2
    p=8
    T=0.01
    V=1
    mu=2
    l=11
 
        
    CT1=tree.Caylee_tree(q, l, V, T, mu)
    CT1.BdG_cycle()
    
    effCT1=tree.effective_Caylee_HL(l,V,T,mu)
    effCT1.BdG_cycle()

    hypersample1=hbdg.centered_HL(l)
    h_sample1=hbdg.HyperBdG(hypersample1,V,T,mu)
    h_sample1.BdG_cycle()
    r_Delta1, r_sigma1=h_sample1.get_radial_Delta()
 
    mu=0
    CT2=tree.Caylee_tree(q, l, V, T, mu)
    CT2.BdG_cycle()
    
    effCT2=tree.effective_Caylee_HL(l,V,T,mu,p=7)
    effCT2.BdG_cycle()
    
    hypersample2=hbdg.centered_HL_hypertiling(7,3,l)
    h_sample2=hbdg.HyperBdG(hypersample2,V,T,mu)
    h_sample2.BdG_cycle()
    r_Delta2, r_sigma2=h_sample2.get_radial_Delta()
    

    
    
    "Plotting"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=10 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    plt.suptitle(r"Radial $\Delta$", fontsize=32, y=1.03)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=24)
    ax1.set_ylabel(r'$\Delta$',fontsize=26)
    ax1.set_xlabel(r'Distance from the center',fontsize=26)
    ax1.plot(r_Delta1, label='\{8,3\} lattice', linewidth=1.9,color='royalblue')
    ax1.plot(r_sigma1, label="{\{8,3\} lattice, $\sigma$", linewidth=1.9,color='coral')
    ax1.plot(effCT1.Delta, linewidth=1.9, label="effective Cayley tree", color='teal')
    ax1.plot(CT1.Delta, linewidth=1.9, label="Cayley tree", color='slategray')
    ax1.legend(fontsize=22)
    ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=5)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=24)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'Distance from the center',fontsize=26)
    ax2.plot(r_Delta2, label='\{7,3\} lattice', linewidth=1.9,color='royalblue')
    ax2.plot(r_sigma2, label="{\{7,3\} lattice, $\sigma$", linewidth=1.9,color='coral')
    ax2.plot(effCT2.Delta, linewidth=1.9, label="effective Cayley tree", color='teal')
    ax2.plot(CT2.Delta, linewidth=1.9, label="Cayley tree", color='slategray')
    ax2.legend(fontsize=22)
    ax2.set_box_aspect(1)
    
    ax1.text(0.05,1.05,'a)', transform=ax1.transAxes, fontsize=28, fontstyle='oblique')
    ax2.text(0.05,1.05,'b)', transform=ax2.transAxes, fontsize=28, fontstyle='oblique')

    #plt.show()
    filename="effective_model_different_cases.pdf"
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    
    return 

def main():
    #plot_DoS_phasediag()
    #plot_hyper_lattices()
    #plot_slice_phase_diagram()
    #plot_profile_on_effective_Cayley_tree()
    #plot_comparison_profiles()
    #plot_various_M()
    compare_effective_trees()    


main()