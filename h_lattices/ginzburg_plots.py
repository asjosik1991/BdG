import numpy as np
import hyper_aux.aux_functions as haux
import Hyperbolic_BdG as hbdg
import effective_models.Cayley_tree as tree
import matplotlib.patches as mpatches

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

from hypertiling import HyperbolicGraph, GraphKernels, HyperbolicTiling
from hypertiling.graphics.plot import plot_tiling
#from hypertiling.kernel.GRG_util import plot_graph
import networkx as nx
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

import matplotlib.cm as cmap
palette = ["#81b29a", "#f2cc8f", "#e07a5f"]

def plot_graph(adjacent_matrix, center_coords, p, ax_plot, colors=[]):
    """
    Plot a network of the connections

    Parameters
    ----------
    adjacent_matrix : List[List[int]]
        Matrix storing the neighboring relations
    center_coords : np.array[n]
        Positions of the node coords as complex
    p : int, optional
        Number of edges of a single polygon in the tiling, default is rotational symmetry

    Returns
    -------
    void
    """
    graph = nx.Graph()
    for y in range(len(adjacent_matrix)):
        if y >= center_coords.shape[0]:
            sector = (y - 1) // (center_coords.shape[0] - 1)
            index = (y - 1) % (center_coords.shape[0] - 1)
            index += 1
            rot = center_coords[index] * np.exp(1j * sector * np.pi * 2 / p)
            x_ = np.real(rot)
            y_ = np.imag(rot)
        else:
            x_ = np.real(center_coords[y])
            y_ = np.imag(center_coords[y])

        #if colors:
        graph.add_node(y, pos=(x_, y_), node_color='xkcd:cobalt')
        #else:
        #    graph.add_node(y, pos=(x_, y_))

    for y, row in enumerate(adjacent_matrix):
        for index in row:
            if index >= len(adjacent_matrix):
                print(f"Skip: {y} -> {index}")
                continue
            graph.add_edge(y, index)

    nx.draw_networkx(graph, pos=nx.get_node_attributes(graph, 'pos'),
                     node_color=list(nx.get_node_attributes(graph, 'node_color').values()),ax=ax_plot,node_size=15, node_shape='.',with_labels=False)
    return


def generate_color_palette(colormap_name, num_samples):
    cmap = plt.get_cmap(colormap_name)  # Get the colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_samples)]  # Generate color samples
    hex_colors = [mpl.colors.to_hex(color) for color in colors]  # Convert to hex
    return hex_colors

disc2stripe = np.vectorize(lambda z: 2 / np.pi * np.log((1 + z) / (1 - z)))
stripe2ring = np.vectorize(lambda z, k, delta: np.exp(2 * np.pi * 1j * (z + 1j) / (k * delta)))

def plot_conformal_tiling(ax, tiling, k, nlayers, delta=1.845, squash=True, wrap=True):
    """
    Draws the tiling on the given axis (ax) with parameter k.
    Default parameters correspond to (3,7) tiling
    """

    fund_region = [-delta/2, delta/2]

    if wrap and not squash:
        squash = False

    for i in range(k):
        for idx, pgon in enumerate(tiling):

            if squash:
                pgonn = disc2stripe(pgon)

                # Exclude polygons outside the fundamental region
                if np.real(pgonn[0]) < fund_region[0] or np.real(pgonn[0]) > fund_region[1]:
                    continue

                # Replicate along the stripe
                pgonn += i * delta

            else:
                pgonn = pgon

            # Color by reflection level
            poly_layer = tiling.get_reflection_level(idx)
            #poly_layer=tiling.get_layer(idx)
            facecolor = generate_color_palette('Blues', nlayers + 5)[poly_layer]
            #facecolor = 'white'


            # Apply second conformal transformation
            if wrap:
                pgonn = stripe2ring(pgonn, k, delta)

            # Draw polygon
            patch = mpl.patches.Polygon(np.array([(np.real(e), np.imag(e)) for e in pgonn[1:]]),
                                        facecolor=facecolor, edgecolor="black", linewidth=0.15)
            ax.add_patch(patch)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.9), dpi=1000, sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    
    ax1.plot(energy_range, dos_array,linewidth=1.1,color='royalblue')
    ax1.locator_params(nbins=5)
    ax1.set_xlabel(r'$s$', fontsize=8,labelpad=0)
    ax1.set_ylabel(r'$\rho(s)$', fontsize=8,labelpad=3)
    ax1.tick_params(labelsize=6)
    ax1.set_box_aspect(1)
    ax1.set_title("DoS", fontsize=10)
    
    pcm=ax2.imshow(field, vmin=field.min(), vmax=field.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
    ax2.locator_params(nbins=5)
    ax2.set_xlabel(r'$\mu$', fontsize=8,labelpad=0)
    ax2.set_ylabel(r'$T$', fontsize=8,labelpad=0)
    ax2.tick_params(labelsize=6)
    ax2.set_box_aspect(1)
    ax2.set_title("Phase diagram", fontsize=10)
    #ax2.set_title('b)', fontfamily='serif', loc='left', fontsize=26, y=1.15,x=-0.2)
    cbar=fig.colorbar(pcm, ax=ax2, shrink=0.9, location='top',pad=0)
    cbar.set_label(label=r'$\bar\Delta$',fontsize=8, rotation=0,x=1.01,labelpad=-6)
    cbar.ax.tick_params(labelsize=6)    
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax1.text(-0.1,1.11,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(-0.12,1.11,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')


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
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 4), layout='constrained',dpi=1000)
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=10 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    
    axs[0,0].plot(r_Delta1, label="{\{8,3\} lattice, $\Delta$", linewidth=1.1,color='royalblue')
    axs[0,0].plot(r_sigma1, label="{\{8,3\} lattice, $\sigma$", linewidth=1.1,color='coral')
    axs[0,0].plot(r_CDelta1, label="Cayley tree, $\Delta$", linewidth=1.1, color='slategray')

    axs[0,0].locator_params(nbins=5)
    axs[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[0,0].set_xlabel(r'Distance from the center', fontsize=22)
    axs[0,0].set_ylabel(r'$\Delta$', fontsize=8,labelpad=1)
    axs[0,0].tick_params(labelsize=8)
    axs[0,0].set_box_aspect(1)
    axs[0,0].set_title("8 shells", fontsize=10)
    axs[0,0].legend(fontsize=8)
    
    axs[1,0].plot(r_Delta2, label="{\{8,3\} lattice, $\Delta$", linewidth=1.1,color='royalblue')
    axs[1,0].plot(r_sigma2, label="{\{8,3\} lattice, $\sigma$", linewidth=1.1,color='coral')
    axs[1,0].plot(r_CDelta2, label="Cayley tree, $\Delta$", linewidth=1.1, color='slategray')

    axs[1,0].locator_params(nbins=5)
    axs[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1,0].set_xlabel(r'Distance from the center', fontsize=8)
    axs[1,0].set_ylabel(r'$\Delta$', fontsize=10,labelpad=1)
    axs[1,0].tick_params(labelsize=8)
    axs[1,0].set_box_aspect(1)
    axs[1,0].set_title("11 shells", fontsize=10)
    #axs[1,0].legend(fontsize=8)

    
    colormap=plt.cm.plasma
    G1 = nx.from_numpy_array(h_sample1.lattice_H)
    nx.draw(G1,pos=nx.shell_layout(G1,nlist=h_sample1.lattice_sample.shell_list,rotate=0),ax=axs[0,1],node_color=h_sample1.Delta, node_size=20, node_shape='.',cmap=colormap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(h_sample2.Delta), vmax=max(h_sample2.Delta)))
    cbar1=fig.colorbar(sm,ax=axs[0,1],shrink=0.7)
    cbar1.ax.tick_params(labelsize=8)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar1.locator = tick_locator
    cbar1.set_label("$\Delta$", fontsize=8, rotation=0, y=1.11,labelpad=-15)
    cbar1.update_ticks()
    axs[0,1].set_title("8 shells", fontsize=10)

    G2 = nx.from_numpy_array(h_sample2.lattice_H)
    nx.draw(G2,pos=nx.shell_layout(G2,nlist=h_sample2.lattice_sample.shell_list,rotate=0),ax=axs[1,1],node_color=h_sample2.Delta, node_size=20, node_shape='.',cmap=colormap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(h_sample2.Delta), vmax=max(h_sample2.Delta)))
    cbar2=fig.colorbar(sm,ax=axs[1,1],shrink=0.7)
    cbar2.ax.tick_params(labelsize=8)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar2.locator = tick_locator
    cbar2.set_label("$\Delta$", fontsize=8, rotation=0, y=1.11,labelpad=-15)
    cbar2.update_ticks()
    axs[1,1].set_title("11 shells", fontsize=10)
    
    axs[0,0].text(-0.1,1.08,'a)', transform=axs[0,0].transAxes, fontsize=10, fontstyle='oblique')
    axs[0,1].text(-0.02,1.08,'b)', transform=axs[0,1].transAxes, fontsize=10, fontstyle='oblique')
    axs[1,0].text(-0.1,1.08,'c)', transform=axs[1,0].transAxes, fontsize=10, fontstyle='oblique')
    axs[1,1].text(-0.02,1.08,'d)', transform=axs[1,1].transAxes, fontsize=10, fontstyle='oblique')
    axs[0,0].set_box_aspect(1)
    axs[1,0].set_box_aspect(1)
    axs[0,1].set_box_aspect(1)
    axs[1,1].set_box_aspect(1)  

    #plt.show()
    
    filename="8_3_lattices_Delta.pdf"
    plt.savefig(filename)
    plt.close()

def plot_slice_phase_diagram():   #Fig.3
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
    T_array=np.linspace(0.01,0.2,100)
    r_CDelta=[]
    h_Delta=[]
    for T in T_array:
        CT=tree.Caylee_tree(q, l, V, T, mu, Delta=r_CDelta)
        CT.BdG_cycle()
        r_CDelta=CT.Delta
        
        h_sample=hbdg.HyperBdG(hypersample,V,T,mu, Delta=h_Delta)
        h_sample.BdG_cycle()
        r_Delta, r_sigma=h_sample.get_radial_Delta()
        h_Delta=h_sample.Delta
        
        Delta_bulk.append(r_Delta[0])
        Delta_edge.append(r_Delta[-1])
        CDelta_bulk.append(r_CDelta[0])
        CDelta_edge.append(r_CDelta[-1])
    
    
    "Plotting"
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(3.4,2.5),dpi=1000,layout='constrained')
    ax.locator_params(nbins=5)
    ax.set_aspect('auto')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    plt.xlabel(r'$T$',fontsize=8,labelpad=1)
    plt.plot(T_array,Delta_edge, label='\{8,3\} lattice, edge', linewidth=1.1,color='royalblue')
    plt.plot(T_array,Delta_bulk, label="\{8,3\} lattice, center", linewidth=1.1,color='firebrick')
    plt.plot(T_array,CDelta_edge, label='Cayley tree, edge', linewidth=1.1,color='slategray')
    plt.plot(T_array,CDelta_bulk, label="Cayley tree, center", linewidth=1.1,color='orchid')
    plt.legend(fontsize=8)

    plt.title(r"Slice of the phase diagram", fontsize=12, y=1.02)

    #plt.show()
    filename="mu_slice_hyplat_tree.pdf"
    plt.savefig(filename)
    plt.close()

def plot_various_M():       #Fig.4

    "Calculation"
    q=2
    p=8
    T=0.01
    V=1
    mu=0
    l_array=np.array([4,5,6,7,8,9,10,11])
    #l_array=np.array([4,5,6,7])

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4,2.2),dpi=1000, sharex=True, sharey=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=0)
    ax1.set_xlabel(r'$M$',fontsize=8,labelpad=0)
    ax1.plot(l_array,Delta_edge1, label='\{8,3\} lattice, edge', linewidth=1.1,color='royalblue')
    ax1.plot(l_array,Delta_bulk1, label="\{8,3\} lattice, center", linewidth=1.1,color='firebrick')
    ax1.plot(l_array,CDelta_edge1, label='Cayley tree, edge', linewidth=1.1,color='slategray')
    ax1.plot(l_array,CDelta_bulk1, label="Cayley tree, center", linewidth=1.1,color='orchid')
    #ax1.legend(fontsize=12)
    #ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=5)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'$M$',fontsize=8,labelpad=1)
    ax2.plot(l_array,Delta_edge2, label='\{8,3\} lattice, edge', linewidth=1.1,color='royalblue')
    ax2.plot(l_array,Delta_bulk2, label="\{8,3\} lattice, center", linewidth=1.1,color='firebrick')
    ax2.plot(l_array,CDelta_edge2, label='Cayley tree, edge', linewidth=1.1,color='slategray')
    ax2.plot(l_array,CDelta_bulk2, label="Cayley tree, center", linewidth=1.1,color='orchid')
    ax2.legend(fontsize=8)
    #ax2.set_box_aspect(1)
    
    ax1.text(0,1.03,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(0,1.03,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')

    plt.suptitle(r"Edge and center $\Delta$", fontsize=12)

    #plt.show()
    filename="various_M.pdf"
    plt.savefig(filename)
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
    fig, ax = plt.subplots(figsize=(3.4,2.5),dpi=1000,layout='constrained')
    ax.locator_params(nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    plt.xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    plt.title(r"Effective Cayley tree", fontsize=12,y=1.02)
    plt.xlim(xmax=l+2)
    plt.plot(CT.Delta, linewidth=1.1,color='teal')
    
    
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
    fig, ax = plt.subplots(figsize=(3.4,2.5),dpi=1000,layout='constrained')
    ax.locator_params(nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    plt.xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    plt.title(r"Radial $\Delta$", fontsize=12,y=1.02)
    
    plt.plot(r_Delta, label='\{8,3\} lattice', linewidth=1.1,color='royalblue')
    plt.plot(effCT.Delta, linewidth=1.1, label="Cayley tree approximation", color='teal')
    plt.plot(CT.Delta, linewidth=1.1, label="Cayley tree", color='slategray')
    plt.legend(fontsize=8)
    
    #plt.show()
    filename="Radial_Deltas.pdf"
    plt.savefig(filename)
    plt.close()
    
    return

def compare_effective_trees():       #Fig.7

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4,2.1),dpi=1000, sharex=True, sharey=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=1/ 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    ax1.set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    ax1.plot(r_Delta1, label=r'$\bar\Delta$', linewidth=1.1,color='royalblue')
    ax1.plot(r_sigma1, label=r"$\sigma$", linewidth=1.1,color='coral')
    ax1.plot(effCT1.Delta, linewidth=1.1, label="Cayley tree approximation", color='teal')
    ax1.plot(CT1.Delta, linewidth=1.1, label="Cayley tree", color='slategray')
    #ax1.legend(fontsize=8)
    ax1.set_title(r"a)  \{8,3\} lattice, $\mu=2$",fontsize=10)
    #ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=5)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    ax2.plot(r_Delta2, linewidth=1.1,color='royalblue')
    ax2.plot(r_sigma2, linewidth=1.1,color='coral')
    ax2.plot(effCT2.Delta, linewidth=1.1, color='teal')
    ax2.plot(CT2.Delta, linewidth=1.1, color='slategray')
    ax2.set_title(r"b)  \{7,3\} lattice, $\mu=0$",fontsize=10)
    #ax2.set_box_aspect(1)
    fig.legend(fontsize=8,loc=(0.2,0.5))
    # ax1.text(0,1.03,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    # ax2.text(0,1.03,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')

    #plt.show()
    filename="effective_model_different_cases.pdf"
    plt.savefig(filename)
    plt.close()
    
    return 

def hyperbolic_sketch():
    
    fig = plt.figure(figsize=(3.4, 1.9),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=0/ 72, h_pad=2/ 72, hspace=0, wspace=0)
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[2,1], height_ratios=[1, 1], figure=fig)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    ax_a = fig.add_subplot(gs[:, 0])  # spans both rows on the left
    ax_b = fig.add_subplot(gs[0, 1])  # top-right
    ax_c = fig.add_subplot(gs[1, 1])  # bottom-right
    
    # ----- Panel (a): draw the Cayley tree -----
    # hypersample1=hbdg.centered_HL(4) 
    # h_sample1=hbdg.HyperBdG(hypersample1,1,1,0)
    # h_sample1.BdG_cycle()
    
    # #ax_a.set_title("Cayley tree")
    # ax_a.set_axis_off()
    # colormap=plt.cm.plasma
    # G1 = nx.from_numpy_array(h_sample1.lattice_H)
    # nx.draw(G1,pos=nx.shell_layout(G1,nlist=h_sample1.lattice_sample.shell_list,rotate=0),ax=ax_a,node_color=h_sample1.Delta, node_size=20, node_shape='.',cmap=colormap)

    nlayers = 6
    tiling = HyperbolicTiling(8, 3, nlayers, kernel="GR", mangle=0)
    plot_conformal_tiling(ax_a, tiling, 1, nlayers, squash=False, wrap=False)
    ax_a.set_title(r'$\{8,3\}$ lattice',fontsize=10)
    # panel tag
   # ax_a.text(0.01, 0.98, "(a)", transform=ax_a.transAxes, ha="left", va="top")
    
    # ----- Panels (b) and (c): simple function plots -----
    x = np.linspace(0, 2, 400)
    
    # (b) y = e^x
    ax_b.plot(x, 1+np.exp(2*x),color='royalblue',linewidth=1.1)
    ax_b.set_ylim(ymin=-8,ymax=51)

    #ax_b.set_ylabel(r'$\Delta$',fontsize=8)
    ax_b.set_xticks([])  # Remove x-axis ticks
    ax_b.set_yticks([])  # Remove y-axis ticks
    ax_b.set_title(r'$\Delta$',fontsize=8)
    #ax_b.set_title("y = exp(x)")
    #ax_b.set_xlabel("x")
    #ax_b.set_ylabel("y")
    #ax_b.text(0.01, 0.98, "(b)", transform=ax_b.transAxes, ha="left", va="top")
    
    # (c) y = e^{-x}
    ax_c.plot(x, 1-np.exp(2*x),color='royalblue',linewidth=1.1)
    ax_c.set_ylim(ymin=-50,ymax=10)
    ax_c.set_xticks([])  # Remove x-axis ticks
    ax_c.set_yticks([])  # Remove y-axis ticks
    #ax_c.set_ylabel(r'$\Delta$',fontsize=8)
    ax_c.set_xlabel(r'$\rho$',fontsize=8)


    #ax_c.set_title("y = exp(-x)")
    #ax_c.set_xlabel("x")
    #ax_c.set_ylabel("y")
    #ax_c.text(0.01, 0.98, "(c)", transform=ax_c.transAxes, ha="left", va="top")
    
    ax_b.set_box_aspect(1)
    ax_c.set_box_aspect(1)
    
    ax_a.set_xlim(-1.1, 1.1)
    ax_a.set_ylim(-1.1, 1.1)
    ax_a.set_box_aspect(1)
    ax_a.set_axis_off()
    #fig.patch.set_facecolor('#F9F9F9')
    ax_a.text(0.05,1.05,'a)', transform=ax_a.transAxes, fontsize=10, fontstyle='oblique')
    ax_b.text(-0.2,1.11,'b)', transform=ax_b.transAxes, fontsize=10, fontstyle='oblique')
    
    arrow = mpatches.FancyArrowPatch((0, 0), (0.6, 0),
                                 mutation_scale=11, facecolor='xkcd:sea blue',edgecolor='xkcd:grey blue',lw=0.5)
    ax_a.add_patch(arrow)
    
    ax_a.text(0.15,0.07,r'$\rho$', fontsize=10, fontstyle='oblique')
    
    #plt.show()
    filename="hyperbolic_sketch.pdf"
    plt.savefig(filename)
    plt.close()

    return

def hyper_tree_comparison():
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.1), dpi=1000, sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    
    G1 = HyperbolicGraph(3, 8, 7, kernel = GraphKernels.GenerativeReflectionGraph)
    nbrs = G1.get_nbrs_list()  # get neighbors
    crds = G1.center_coords    # get coordinates of cell centers
    p = G1.p                   # lattice parameter p
    
    # color nodes by layer
    #colors = [palette[G1.get_reflection_level(i) % len(generate_color_palette('Blues', 5))] for i in range(G1.length)]
    #colors=generate_color_palette('Blues', 12)
    plot_graph(nbrs, crds, p, ax1)

    
    G2 = HyperbolicGraph(3, 15, 7, kernel = GraphKernels.GenerativeReflectionGraph)
    nbrs = G2.get_nbrs_list()  # get neighbors
    crds = G2.center_coords    # get coordinates of cell centers
    p = G2.p                   # lattice parameter p
    
    # color nodes by layer
    #colors = [palette[G2.get_reflection_level(i) % len(generate_color_palette('Blues', 5))] for i in range(G2.length)]
    
    plot_graph(nbrs, crds, p, ax2)
    
    ax1.set_title(r'$\{8,3\}$ lattice',fontsize=10)
    ax2.set_title(r'Cayley tree',fontsize=10)
    ax1.text(0.1,1.05,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(0.1,1.05,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')

    ax1.set_box_aspect(1)
    ax1.set_axis_off()
    
    
    ax2.set_box_aspect(1)
    ax2.set_axis_off()
    
    #plt.show()
    filename="hyperlat_trees_example.pdf"
    plt.savefig(filename)
    plt.close()

    return



def main():
    #plot_DoS_phasediag()                       #Fig.1
    #plot_hyper_lattices()                      #Fig.2
    #plot_slice_phase_diagram()                 #Fig.3
    #plot_various_M()                           #Fig.4
    #plot_profile_on_effective_Cayley_tree()    #Fig.5
    #plot_comparison_profiles()                 #Fig.6
    #compare_effective_trees()                  #Fig.7
    #hyperbolic_sketch()
    hyper_tree_comparison()

main()