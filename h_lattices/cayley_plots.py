import numpy as np
import hyper_aux.aux_functions as haux
from matplotlib import rc
import networkx as nx
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
import effective_models.Cayley_tree as tree
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cmap
from matplotlib.patches import Arc
from functools import lru_cache

def plot_DoS_phasediag():     #Fig.1
   
    energy_range=np.linspace(-4.0, 4.0, num=150)
    dos_array=[]
    for s in energy_range:
        dos_array.append(haux.bethe_dos3(s))
    
    V_array=[1]
    mu_array=energy_range
    T_array=np.linspace(0.001, 0.015, num=200)
    haux.calculate_exactdiagram(haux.bethe_dos3, V_array, mu_array, T_array)
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
 
   # plt.show()
    filename="bethe_bulk_example.pdf"
    plt.savefig(filename)
    plt.close()


def plot_phases():       #Fig.2

    "Calculation"
    q=2
    p=8
    T=0.01
    V=1
    mu=0
    l1=10
    l2=100
    #l_array=np.array([4,5,6,7])
    T_array_bulk=np.linspace(0.001,0.014,50)
    
    
    V_array=[1]
    mu_array=[0]
    haux.calculate_exactdiagram(haux.bethe_dos3, V_array, mu_array, T_array_bulk)
    hyper=haux.load_exactdiagram()
    if len(hyper['V'])==1:
        V=hyper['V'][0]
        x=hyper['mu']
        y=hyper['T']
        Deltas=hyper['Deltas']
        # print(x)
        # print(y)
        # print(Deltas)
        bulk_Delta_Bethe=np.zeros((len(y),len(x)))      

        for i in range(len(y)):
            for j in range(len(x)):
                bulk_Delta_Bethe[i,j]=Deltas[(V,y[i],x[j])]
    print(bulk_Delta_Bethe)

    T_array=np.linspace(0.001,0.2,50)
    CDelta_bulk1=[]
    CDelta_edge1=[]
    CDelta_bulk2=[]
    CDelta_edge2=[]
      
    r_CDelta1=[]
    r_CDelta2=[]
    for T in T_array:
        CT1=tree.Caylee_tree(q, l1, V, T, mu, Delta=r_CDelta1)
        CT1.BdG_cycle()
        r_CDelta1=CT1.Delta

        CDelta_bulk1.append(r_CDelta1[0])
        CDelta_edge1.append(r_CDelta1[-1])
        
        CT2=tree.Caylee_tree(q, l2, V, T, mu, Delta=r_CDelta2)
        CT2.BdG_cycle()
        r_CDelta2=CT2.Delta

        CDelta_bulk2.append(r_CDelta2[0])
        CDelta_edge2.append(r_CDelta2[-1])
    
    lb1=100
    lb2=200
    lb3=300
    bCDelta_bulk1=[]
    bCDelta_edge1=[]
    bCDelta_bulk2=[]
    bCDelta_edge2=[]
    bCDelta_bulk3=[]
    bCDelta_edge3=[]
      
    br_CDelta1=[]
    br_CDelta2=[]
    br_CDelta3=[]
    for T in T_array_bulk:
        bCT1=tree.Caylee_tree(q, lb1, V, T, mu, Delta=br_CDelta1)
        bCT1.BdG_cycle()
        br_CDelta1=bCT1.Delta

        bCDelta_bulk1.append(br_CDelta1[0])
        bCDelta_edge1.append(br_CDelta1[-1])
        
        bCT2=tree.Caylee_tree(q, lb2, V, T, mu, Delta=br_CDelta2)
        bCT2.BdG_cycle()
        br_CDelta2=bCT2.Delta

        bCDelta_bulk2.append(br_CDelta2[0])
        bCDelta_edge2.append(br_CDelta2[-1])
        
        bCT3=tree.Caylee_tree(q, lb3, V, T, mu, Delta=br_CDelta3)
        bCT3.BdG_cycle()
        br_CDelta3=bCT3.Delta

        bCDelta_bulk3.append(br_CDelta3[0])
        bCDelta_edge3.append(br_CDelta3[-1])
    
    
    
    "Plotting"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7,2.3),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=5)
    #ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=0)
    ax1.set_xlabel(r'$T$',fontsize=8,labelpad=0)

    ax1.plot(T_array,CDelta_edge1, label='edge', linewidth=1.1,color='slategray')
    ax1.plot(T_array,CDelta_bulk1, label="center", linewidth=1.1,color='orchid')
    ax1.legend(fontsize=8)
    ax1.set_title(str(l1)+" shells", fontsize=10)

    #ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=5)
    #ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'$T$',fontsize=8,labelpad=1)

    ax2.plot(T_array,CDelta_edge2, label='edge', linewidth=1.1,color='slategray')
    ax2.plot(T_array,CDelta_bulk2, label="center", linewidth=1.1,color='orchid')
    ax2.set_title(str(l2)+" shells", fontsize=10)
    ax2.legend(fontsize=8)
    #ax2.set_box_aspect(1)
    
    ax3.locator_params(nbins=5)
    #ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax3.set_xlabel(r'$T$',fontsize=8,labelpad=1)

    ax3.plot(T_array_bulk,bCDelta_bulk1, label=str(lb1)+'shells', linewidth=1.1,color='lightcoral')
    ax3.plot(T_array_bulk,bCDelta_bulk2, label=str(lb2)+'shells', linewidth=1.1,color='indianred')
    ax3.plot(T_array_bulk,bCDelta_bulk3, label=str(lb3)+'shells', linewidth=1.1,color='firebrick')

    ax3.plot(T_array_bulk,bulk_Delta_Bethe[:,0], label='Bethe lattice', linewidth=1.1,color='deeppink')
    ax3.set_title("Bulk comparison", fontsize=10)

    ax3.legend(fontsize=8,loc='lower right')
    #ax2.set_box_aspect(1)
    
    ax1.text(0,1.03,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(0,1.03,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')
    ax3.text(0,1.03,'c)', transform=ax3.transAxes, fontsize=10, fontstyle='oblique')


    #plt.suptitle(r"Edge and center $\Delta$", fontsize=12)

    #plt.show()
    filename="cayley_tree_bulks.pdf"
    plt.savefig(filename)
    plt.close()
    
    return 

def example_states_mu0():       #Fig.3

    "Calculation"
    q=2
    p=8
    T=0.1
    V=1
    mu=0
    l1=8
    l2=100
 
        
    CT1=tree.Caylee_tree(q, l1, V, T, mu)
    CT1.BdG_cycle()
    
    CT2=tree.Caylee_tree(q, l2, V, T, mu)
    CT2.BdG_cycle()
    
    
    "Plotting"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4,2),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=2/ 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    ax1.set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    ax1.plot(CT1.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    ax1.set_title("a) \enspace\enspace\enspace\enspace" +str(l1)+" shells",fontsize=10,loc='left')
    #ax1.set_box_aspect(1)
    
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    #ax2.set_yticks([0.1, 0.01, 0.001])
    #ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax2.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    ax2.semilogy(CT2.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax2.legend(fontsize=8)
    ax2.set_title("b) \enspace\enspace\enspace\enspace" +str(l2)+" shells",fontsize=10,loc='left')
    ax2.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))    #ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    #ax2.set_box_aspect(1)
    #print(CT2.Delta[0])
    # ax1.text(0,1.03,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    # ax2.text(0,1.03,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')

    #plt.show()
    filename="example_states_mu0.pdf"
    plt.savefig(filename)
    plt.close()
    
    return


def example_states_different_mu():       #Fig.4

    "Calculation"
    q=2
    p=8
    T=0.001
    V=1
    l=100
    mu1=0.5
    mu2=1.1
    mu3=2.3
    mu4=2.76
 
        
    CT1=tree.Caylee_tree(q, l, V, T, mu1)
    CT1.BdG_cycle()
    
    CT2=tree.Caylee_tree(q, l, V, T, mu2)
    CT2.BdG_cycle()
    
    CT3=tree.Caylee_tree(q, l, V, T, mu3)
    CT3.BdG_cycle()
    
    CT4=tree.Caylee_tree(q, l, V, T, mu4)
    CT4.BdG_cycle()
    
    
    "Plotting"
    fig, axs = plt.subplots(2, 2, figsize=(3.4,4),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=3/ 72, h_pad=4 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    axs[0,0].locator_params(nbins=5)
    axs[0,0].xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    axs[0,0].tick_params(labelsize=8)
    axs[0,0].set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    axs[0,0].set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    axs[0,0].plot(CT1.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    axs[0,0].set_title("a) \enspace\enspace\enspace\enspace$\mu=" +str(mu1)+"$",fontsize=10,loc='left')
    #ax1.set_box_aspect(1)
    


    axs[0,1].locator_params(nbins=5)
    axs[0,1].xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    axs[0,1].tick_params(labelsize=8)
    axs[0,1].set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    axs[0,1].set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    axs[0,1].plot(CT2.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    axs[0,1].set_title("b) \enspace\enspace\enspace\enspace$\mu=" +str(mu2)+"$",fontsize=10,loc='left')
    #ax1.set_box_aspect(1)
    
    axs[1,0].locator_params(nbins=5)
    axs[1,0].xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    axs[1,0].tick_params(labelsize=8)
    axs[1,0].set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    axs[1,0].set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    axs[1,0].plot(CT3.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    axs[1,0].set_title("c) \enspace\enspace\enspace\enspace$\mu=" +str(mu3)+"$",fontsize=10,loc='left')
    
    
    axs[1,1].locator_params(nbins=5)
    axs[1,1].xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    axs[1,1].tick_params(labelsize=8)
    axs[1,1].set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    axs[1,1].set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    axs[1,1].plot(CT4.Delta, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    axs[1,1].set_title("d) \enspace\enspace\enspace\enspace$\mu=" +str(mu4)+"$",fontsize=10,loc='left')

    #plt.show()
    filename="example_states_varmu.pdf"
    plt.savefig(filename)
    plt.close()
    
    return

def plot_phases_nonzero_mu():       #Fig.2

    "Calculation"
    q=2
    p=8
    T=0.01
    V=1
    mu1=1.1
    mu2=2.76
    l=100
    #l_array=np.array([4,5,6,7])

    T1_array=np.linspace(0.0001,0.012,50)
    T2_array=np.linspace(0.0001,0.006,50)
    
    V_array=[1]
    mu_array=[0]
    haux.calculate_exactdiagram(haux.bethe_dos3, V_array, mu_array, T1_array)
    hyper=haux.load_exactdiagram()
    if len(hyper['V'])==1:
        V=hyper['V'][0]
        x=hyper['mu']
        y=hyper['T']
        Deltas=hyper['Deltas']
        # print(x)
        # print(y)
        # print(Deltas)
        bulk_Delta_Bethe1=np.zeros((len(y),len(x)))      

        for i in range(len(y)):
            for j in range(len(x)):
                bulk_Delta_Bethe1[i,j]=Deltas[(V,y[i],x[j])]
    haux.calculate_exactdiagram(haux.bethe_dos3, V_array, mu_array, T1_array)
    hyper=haux.load_exactdiagram()
    if len(hyper['V'])==1:
        V=hyper['V'][0]
        x=hyper['mu']
        y=hyper['T']
        Deltas=hyper['Deltas']
        # print(x)
        # print(y)
        # print(Deltas)
        bulk_Delta_Bethe2=np.zeros((len(y),len(x)))      

        for i in range(len(y)):
            for j in range(len(x)):
                bulk_Delta_Bethe2[i,j]=Deltas[(V,y[i],x[j])]
    #print(bulk_Delta_Bethe)
    
    
    
    CDelta_bulk1=[]
    CDelta_edge1=[]
    CDelta_bulk2=[]
    CDelta_edge2=[]
      
    r_CDelta1=[]
    r_CDelta2=[]
    for T in T1_array:
        CT1=tree.Caylee_tree(q, l, V, T, mu1, Delta=r_CDelta1)
        CT1.BdG_cycle()
        r_CDelta1=CT1.Delta

        CDelta_bulk1.append(r_CDelta1[0])
        CDelta_edge1.append(np.max(r_CDelta1))
    
    for T in T2_array:

        CT2=tree.Caylee_tree(q, l, V, T, mu2, Delta=r_CDelta2)
        CT2.BdG_cycle()
        r_CDelta2=CT2.Delta

        CDelta_bulk2.append(r_CDelta2[0])
        CDelta_edge2.append(np.max(r_CDelta2))
        
    
    "Plotting"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4,2),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=3)
    #ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=0)
    ax1.set_xlabel(r'$T$',fontsize=8,labelpad=0)

    ax1.plot(T1_array,CDelta_edge1, label=r'$\Delta_\mathrm{max}$', linewidth=1.1,color='slategray')
    ax1.plot(T1_array,CDelta_bulk1, label="center", linewidth=1.1,color='orchid')
    ax1.plot(T1_array,bulk_Delta_Bethe1[:,0], label='Bethe lattice', linewidth=1.1,color='deeppink')

    ax1.legend(fontsize=8)
    ax1.set_title("$\mu="+str(mu1)+"$", fontsize=10)

    #ax1.set_box_aspect(1)
    
    ax2.locator_params(nbins=3)
    #ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.tick_params(labelsize=8)
    #ax2.set_ylabel(r'$\Delta$',fontsize=26)
    ax2.set_xlabel(r'$T$',fontsize=8,labelpad=1)

    ax2.plot(T2_array,CDelta_edge2, label=r'$\Delta_\mathrm{max}$', linewidth=1.1,color='slategray')
    ax2.plot(T2_array,CDelta_bulk2, label="center", linewidth=1.1,color='orchid')
    ax2.plot(T2_array,bulk_Delta_Bethe2[:,0], label='Bethe lattice', linewidth=1.1,color='deeppink')

    ax2.set_title("$\mu="+str(mu2)+"$", fontsize=10)
    #ax2.legend(fontsize=8)
    #ax2.set_box_aspect(1)

    
    ax1.text(0,1.04,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(0,1.04,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')


    #plt.suptitle(r"Edge and center $\Delta$", fontsize=12)

    #plt.show()
    filename="cayley_tree_center_maxDelta.pdf"
    plt.savefig(filename)
    plt.close()
    
    return

def plot_Cayley_phasediag():     #Fig.1
   
    mu_array=np.linspace(-4, 4, num=150)
    T_array=np.linspace(0.001, 0.05, num=200)
    V=1
    q=2
    l=40
    x=mu_array
    y=T_array
    CDelta_bulk=np.zeros((len(T_array),len(mu_array)))
    CDelta_edge=np.zeros((len(T_array),len(mu_array)))
    for i in range(len(mu_array)):
       r_CDelta=[]
       for j in range(len(T_array)):
        CT=tree.Caylee_tree(q, l, V, T_array[j], mu_array[i], Delta=r_CDelta)
        CT.BdG_cycle()
        r_CDelta=CT.Delta
    
        CDelta_bulk[j,i]=r_CDelta[0]
        CDelta_edge[j,i]=np.max(r_CDelta)
            
    "Plots"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2), dpi=1000, sharex=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    # plt.xticks(fontsize=24)
    # plt.yticks(fontsize=24)
    
    pcm=ax1.imshow(CDelta_bulk, vmin=CDelta_bulk.min(), vmax=CDelta_edge.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
    ax1.locator_params(nbins=5)
    ax1.set_xlabel(r'$\mu$', fontsize=8,labelpad=0)
    ax1.set_ylabel(r'$T$', fontsize=8,labelpad=0)
    ax1.tick_params(labelsize=6)
    ax1.set_box_aspect(1)
    ax1.set_title("Center $\Delta$", fontsize=10)
    
    pcm=ax2.imshow(CDelta_edge, vmin=CDelta_edge.min(), vmax=CDelta_edge.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
    ax2.locator_params(nbins=5)
    ax2.set_xlabel(r'$\mu$', fontsize=8,labelpad=0)
    ax2.set_ylabel(r'$T$', fontsize=8,labelpad=0)
    ax2.tick_params(labelsize=6)
    ax2.set_box_aspect(1)
    ax2.set_title("Maximum $\Delta$", fontsize=10)
    #ax2.set_title('b)', fontfamily='serif', loc='left', fontsize=26, y=1.15,x=-0.2)
    cbar=fig.colorbar(pcm, ax=[ax1,ax2], shrink=0.6, location='top',pad=0.1)
    cbar.set_label(label=r'$\Delta$',fontsize=8, rotation=0,x=1.05,labelpad=-6)
    cbar.ax.tick_params(labelsize=6)    
    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax1.text(-0.05,1.08,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.text(-0.05,1.08,'b)', transform=ax2.transAxes, fontsize=10, fontstyle='oblique')
 
    #plt.show()
    filename="cayley_tree_phase_diags.pdf"
    plt.savefig(filename)
    plt.close()
    
def plot_phase_uniform():

    "Calculation"

    T_array=np.linspace(0.001, 0.02, num=50)
    V=2
    mu=10
    Emin=8
    Emax=12

    Delta_array=[]
    Delta_ini=1
    for T in T_array:
        Delta=haux.BCS_gap_hyper(mu, T, Emin, Emax, Delta_seed=Delta_ini, V=V)
        Delta_array.append(Delta)
        Delta_ini=Delta
    
    
    "Plotting"
    fig, (ax1) = plt.subplots(1, 1, figsize=(3.4,2),dpi=1000, layout='constrained')
    fig.get_layout_engine().set(w_pad=2/ 72, h_pad=0 / 72, hspace=0, wspace=0)
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)

    ax1.locator_params(nbins=5)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax1.tick_params(labelsize=8)
    ax1.set_ylabel(r'$\Delta$',fontsize=8,labelpad=1)
    ax1.set_xlabel(r'Distance from the center',fontsize=8,labelpad=1)
    ax1.plot(T_array, Delta_array, linewidth=1.1, label="Cayley tree", color='royalblue')
    #ax1.legend(fontsize=8)
    #ax1.set_title("a) \enspace\enspace\enspace\enspace" +str(l1)+" shells",fontsize=10,loc='left')
    #ax1.set_box_aspect(1)
    


    plt.show()
    # filename="example_states_mu0.pdf"
    # plt.savefig(filename)
    # plt.close()
    
    return

def symmetry_adapted_states():
    branching=3
    shells=4
    dr=1.0
    node_size=30
    edge_alpha=0.6
    # Build the tree: balanced r-ary tree of height `shells`
    G = nx.balanced_tree(r=branching, h=shells)
    root = 0

    # Shell (layer) for every node by shortest-path distance from root
    dist = nx.single_source_shortest_path_length(G, root)
    max_shell = max(dist.values())

    # Build a rooted view: children(u) exclude parent to keep it a tree outward from root
    def children_of(u, parent):
       for v in G.neighbors(u):
           if v != parent:
               yield v

    # Count leaves in each subtree (used to allocate angular spans nicely)
    from functools import lru_cache
    @lru_cache(None)
    def leaf_count(u, parent):
        kids = list(children_of(u, parent))
        if not kids:
            return 1
        return sum(leaf_count(v, u) for v in kids)

    # Assign angles recursively so that subtrees occupy contiguous angular spans
    pos = {}
    def assign_angles(u, parent, a0, a1):
        theta = 0.5*(a0 + a1)
        r = dist[u] * dr
        pos[u] = (r * np.cos(theta), r * np.sin(theta))
        kids = list(children_of(u, parent))
        if not kids:
            return
        total = sum(leaf_count(v, u) for v in kids)
        a = a0
        for v in kids:
            span = (a1 - a0) * leaf_count(v, u) / total
            assign_angles(v, u, a, a + span)
            a += span

    assign_angles(root, parent=None, a0=0.0, a1=2*np.pi)

    # Prepare figure/axes
    fig, ax = plt.subplots(figsize=(3.4,2.2),dpi=1000, layout='constrained')
    ax.set_aspect('equal')

    # Draw edges first (so nodes sit on top)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=edge_alpha)

    # Draw nodes shell-by-shell with distinct colors
    cmap = matplotlib.colormaps['Pastel1']  # discrete colors
    for s in range(max_shell + 1):
        nodes_s = [n for n, d in dist.items() if d == s]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes_s, node_size=node_size,
            node_color=[cmap(s)], edgecolors='black', linewidths=0.3, ax=ax
        )

    # Concentric shell circles + labels (shell 1..max_shell; root is shell 0)
    for s in range(1, max_shell + 1):
        radius = s * dr
        circle = plt.Circle((0, 0), radius=radius, fill=False, linestyle='--',linewidth=1.9, color=cmap(s))
        ax.add_patch(circle)
        # Label at the top of each circle; tweak offset so text doesn't sit on the line
        ax.text(-0.2+np.cos(0.1*np.pi)*radius, 0.2+np.sin(0.1*np.pi)*radius, r"$S_{"+str(s)+"}$", ha='center', va='bottom',
               fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax.text(-0.2, 0.2, r"$S_{0}$", ha='center', va='bottom',
           fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # Tidy up frame and limits
    R = (max_shell + 0.8) * dr
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.axis('off')
    ax.set_title("Shells of a Cayley tree", fontsize=10)
    #plt.tight_layout()
    #plt.show()
    filename="cayley_shells.png"
    plt.savefig(filename)
    plt.close()
    
    return

def plot_two_branch_wedge_with_colored_arcs(
    shells=6,
    dr=1.0,
    node_size=55,
    edge_alpha=0.6,
    arc_lw=1.6,
    arc_alpha=0.9,
    total_span = 4*math.pi/3,   # ~240°
    center_angle = -math.pi/2,  # grow downward
    left_cmap_name="Blues",
    right_cmap_name="Oranges",
    title="Binary subtree in a wedge"):
    branching = 2
    G = nx.balanced_tree(r=branching, h=shells)
    root = 0

    def children_of(u, parent):
        for v in G.neighbors(u):
            if v != parent:
                yield v

    @lru_cache(None)
    def leaf_count(u, parent):
        kids = list(children_of(u, parent))
        if not kids:
            return 1
        return sum(leaf_count(v, u) for v in kids)

    dist = nx.single_source_shortest_path_length(G, root)
    max_shell = max(dist.values())

    a0 = center_angle - 0.5 * total_span
    a1 = center_angle + 0.5 * total_span

    pos, ang = {}, {}
    def assign_angles(u, parent, left_a, right_a):
        theta = 0.5 * (left_a + right_a)
        r = dist[u] * dr
        pos[u] = (r * math.cos(theta), r * math.sin(theta))
        ang[u] = theta
        kids = list(children_of(u, parent))
        if not kids:
            return
        total = sum(leaf_count(v, u) for v in kids)
        a = left_a
        for v in kids:
            span = (right_a - left_a) * leaf_count(v, u) / total
            assign_angles(v, u, a, a + span); a += span

    assign_angles(root, None, a0, a1)

    nodes_by_shell = {s: [n for n, d in dist.items() if d == s] for s in range(max_shell + 1)}

    root_children = list(children_of(root, None))
    if len(root_children) != 2:
        raise RuntimeError("Expected exactly two outward branches from the root.")
    left_root, right_root = root_children

    def subtree_nodes(start, parent):
        stack = [(start, parent)]; seen = set()
        while stack:
            u, p = stack.pop()
            if u in seen: continue
            seen.add(u)
            for v in children_of(u, p): stack.append((v, u))
        return seen

    left_nodes  = subtree_nodes(left_root,  root)
    right_nodes = subtree_nodes(right_root, root)

    left_cmap  = plt.cm.get_cmap(left_cmap_name,  max_shell + 1)
    right_cmap = plt.cm.get_cmap(right_cmap_name, max_shell + 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    # edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, alpha=edge_alpha)

    # nodes by shell and branch
    for s in range(max_shell + 1):
        if s == 0:
            nx.draw_networkx_nodes(G, pos, nodelist=[root], node_size=node_size+20,
                                   node_color=["white"], edgecolors='black', linewidths=0.8, ax=ax)
            continue
        ns = nodes_by_shell[s]
        left_shell  = [n for n in ns if n in left_nodes]
        right_shell = [n for n in ns if n in right_nodes]
        if left_shell:
            nx.draw_networkx_nodes(G, pos, nodelist=left_shell, node_size=node_size,
                                   node_color=[left_cmap(s)], edgecolors='black', linewidths=0.35, ax=ax)
        if right_shell:
            nx.draw_networkx_nodes(G, pos, nodelist=right_shell, node_size=node_size,
                                   node_color=[right_cmap(s)], edgecolors='black', linewidths=0.35, ax=ax)

    def unwrap(theta, base):
        twopi = 2*math.pi
        while theta < base: theta += twopi
        while theta >= base + twopi: theta -= twopi
        return theta

    # arcs with matching colors
    for s in range(1, max_shell + 1):
        radius = s * dr
        pad = 0.03

        left_shell_nodes = [n for n in nodes_by_shell[s] if n in left_nodes]
        if left_shell_nodes:
            uw = [unwrap(ang[n], a0) for n in left_shell_nodes]
            a_min, a_max = min(uw), max(uw)
            a_min_p = max(a0, a_min - pad); a_max_p = min(a0 + 2*math.pi, a_max + pad)
            ax.add_patch(Arc((0,0), 2*radius, 2*radius, angle=np.degrees(a0),
                             theta1=np.degrees(a_min_p - a0), theta2=np.degrees(a_max_p - a0),
                             linewidth=arc_lw, alpha=arc_alpha, linestyle='-', edgecolor=left_cmap(s)))
            amid = 0.5*(a_min_p + a_max_p)
            ax.text((radius+0.06*dr)*math.cos(amid), (radius+0.06*dr)*math.sin(amid), f"{s}",
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))

        right_shell_nodes = [n for n in nodes_by_shell[s] if n in right_nodes]
        if right_shell_nodes:
            uw = [unwrap(ang[n], a0) for n in right_shell_nodes]
            a_min, a_max = min(uw), max(uw)
            a_min_p = max(a0, a_min - pad); a_max_p = min(a0 + 2*math.pi, a_max + pad)
            ax.add_patch(Arc((0,0), 2*radius, 2*radius, angle=np.degrees(a0),
                             theta1=np.degrees(a_min_p - a0), theta2=np.degrees(a_max_p - a0),
                             linewidth=arc_lw, alpha=arc_alpha, linestyle='--', edgecolor=right_cmap(s)))
            amid = 0.5*(a_min_p + a_max_p)
            ax.text((radius+0.06*dr)*math.cos(amid), (radius+0.06*dr)*math.sin(amid), f"{s}",
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))

    # title + push content upward
    ax.set_title(title, fontsize=11, pad=4)
    R = (max_shell + 0.9) * dr
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, 0.8*R)  # trims unused top space
    ax.axis('off')
    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.04, right=0.96)
    plt.show()


def combined_figure_shells():
    
    branching1=3
    shells1=4
    shells2=3
    dr=1.0
    node_size=30
    edge_alpha=0.6
    # Build the tree: balanced r-ary tree of height `shells`
    G1 = nx.balanced_tree(r=branching1, h=shells1)
    root = 0
    
    #shells=6,
    #dr=1.0,
    #node_size=55,
    #edge_alpha=0.6,
    arc_lw=1.6
    arc_alpha=0.9
    total_span = 4*math.pi/3 # ~240°
    center_angle = -math.pi/2  # grow downward
    left_cmap_name="Blues"
    right_cmap_name="Oranges"
    #title="Binary subtree in a wedge"
    branching2 = 2
    G2 = nx.balanced_tree(r=branching2, h=shells2)
    #root = 0

    def children_of2(u, parent):
        for v in G2.neighbors(u):
            if v != parent:
                yield v

    #@lru_cache(None)
    def leaf_count2(u, parent):
        kids = list(children_of2(u, parent))
        if not kids:
            return 1
        return sum(leaf_count2(v, u) for v in kids)

    dist2 = nx.single_source_shortest_path_length(G2, root)
    max_shell2 = max(dist2.values())

    a0 = center_angle - 0.5 * total_span
    a1 = center_angle + 0.5 * total_span

    pos2, ang = {}, {}
    def assign_angles(u, parent, left_a, right_a):
        theta = 0.5 * (left_a + right_a)
        r = dist2[u] * dr
        pos2[u] = (r * math.cos(theta), r * math.sin(theta))
        ang[u] = theta
        kids = list(children_of2(u, parent))
        if not kids:
            return
        total = sum(leaf_count2(v, u) for v in kids)
        a = left_a
        for v in kids:
            span = (right_a - left_a) * leaf_count2(v, u) / total
            assign_angles(v, u, a, a + span); a += span

    assign_angles(root, None, a0, a1)

    nodes_by_shell = {s: [n for n, d in dist2.items() if d == s] for s in range(max_shell2 + 1)}

    root_children = list(children_of2(root, None))
    if len(root_children) != 2:
        raise RuntimeError("Expected exactly two outward branches from the root.")
    left_root, right_root = root_children

    def subtree_nodes(start, parent):
        stack = [(start, parent)]; seen = set()
        while stack:
            u, p = stack.pop()
            if u in seen: continue
            seen.add(u)
            for v in children_of2(u, p): stack.append((v, u))
        return seen

    left_nodes  = subtree_nodes(left_root,  root)
    right_nodes = subtree_nodes(right_root, root)

    left_cmap  = plt.cm.get_cmap(left_cmap_name,  max_shell2 + 1)
    right_cmap = plt.cm.get_cmap(right_cmap_name, max_shell2 + 1)
    
    

    # Shell (layer) for every node by shortest-path distance from root
    dist = nx.single_source_shortest_path_length(G1, root)
    max_shell = max(dist.values())

    # Build a rooted view: children(u) exclude parent to keep it a tree outward from root
    def children_of1(u, parent):
       for v in G1.neighbors(u):
           if v != parent:
               yield v

    # Count leaves in each subtree (used to allocate angular spans nicely)
    from functools import lru_cache
    @lru_cache(None)
    def leaf_count1(u, parent):
        kids = list(children_of1(u, parent))
        if not kids:
            return 1
        return sum(leaf_count1(v, u) for v in kids)

    # Assign angles recursively so that subtrees occupy contiguous angular spans
    pos1 = {}
    def assign_angles(u, parent, a0, a1):
        theta = 0.5*(a0 + a1)
        r = dist[u] * dr
        pos1[u] = (r * np.cos(theta), r * np.sin(theta))
        kids = list(children_of1(u, parent))
        if not kids:
            return
        total = sum(leaf_count1(v, u) for v in kids)
        a = a0
        for v in kids:
            span = (a1 - a0) * leaf_count1(v, u) / total
            assign_angles(v, u, a, a + span)
            a += span

    assign_angles(root, parent=None, a0=0.0, a1=2*np.pi)

    # Prepare figure/axes
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(3.4,2),dpi=1000, layout='constrained')
    ax1.set_aspect('equal')

    # Draw edges first (so nodes sit on top)
    nx.draw_networkx_edges(G1, pos1, ax=ax1, width=1.0, alpha=edge_alpha)

    # Draw nodes shell-by-shell with distinct colors
    cmap = matplotlib.colormaps['Pastel1']  # discrete colors
    for s in range(max_shell + 1):
        nodes_s = [n for n, d in dist.items() if d == s]
        nx.draw_networkx_nodes(
            G1, pos1, nodelist=nodes_s, node_size=node_size,
            node_color=[cmap(s)], edgecolors='black', linewidths=0.3, ax=ax1
        )

    # Concentric shell circles + labels (shell 1..max_shell; root is shell 0)
    for s in range(1, max_shell + 1):
        radius = s * dr
        circle = plt.Circle((0, 0), radius=radius, fill=False, linestyle='--',linewidth=1.9, color=cmap(s))
        ax1.add_patch(circle)
        # Label at the top of each circle; tweak offset so text doesn't sit on the line
        ax1.text(-0.2+np.cos(0.1*np.pi)*radius, 0.2+np.sin(0.1*np.pi)*radius, r"$S_{"+str(s)+"}$", ha='center', va='bottom',
               fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    ax1.text(-0.2, 0.2, r"$S_{0}$", ha='center', va='bottom',
           fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    

    ax2.set_aspect('equal')
    
    # edges
    nx.draw_networkx_edges(G2, pos2, ax=ax2, width=1.0, alpha=edge_alpha)
    
    # nodes by shell and branch
    for s in range(max_shell2+1):
        if s == 0:
            nx.draw_networkx_nodes(G2, pos2, nodelist=[root], node_size=node_size,
                                   node_color=["white"], edgecolors='black', linewidths=0.8, ax=ax2)
            continue
        ns = nodes_by_shell[s]
        left_shell  = [n for n in ns if n in left_nodes]
        right_shell = [n for n in ns if n in right_nodes]
        if left_shell:
            nx.draw_networkx_nodes(G2, pos2, nodelist=left_shell, node_size=node_size,
                                   node_color=[left_cmap(s)], edgecolors='black', linewidths=0.35, ax=ax2)
        if right_shell:
            nx.draw_networkx_nodes(G2, pos2, nodelist=right_shell, node_size=node_size,
                                   node_color=[right_cmap(s)], edgecolors='black', linewidths=0.35, ax=ax2)

    def unwrap(theta, base):
        twopi = 2*math.pi
        while theta < base: theta += twopi
        while theta >= base + twopi: theta -= twopi
        return theta

    # arcs with matching colors
    for s in range(1, max_shell2+1):
        radius = s * dr
        pad = 0.03

        left_shell_nodes = [n for n in nodes_by_shell[s] if n in left_nodes]
        if left_shell_nodes:
            uw = [unwrap(ang[n], a0) for n in left_shell_nodes]
            a_min, a_max = min(uw), max(uw)
            a_min_p = max(a0, a_min - pad); a_max_p = min(a0 + 2*math.pi, a_max + pad)
            ax2.add_patch(Arc((0,0), 2*radius, 2*radius, angle=np.degrees(a0),
                             theta1=np.degrees(a_min_p - a0), theta2=np.degrees(a_max_p - a0),
                             linewidth=arc_lw, alpha=arc_alpha, linestyle='--', edgecolor=left_cmap(s)))
            amid = 0.5*(a_min_p + a_max_p)
            ax2.text((1.2*radius+0.06*dr)*math.cos(amid-0.3)+0.5, (1.2*radius+0.06*dr)*math.sin(amid-0.3)-0.7, r"$S_{1,"+str(s)+"}$",
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))

        right_shell_nodes = [n for n in nodes_by_shell[s] if n in right_nodes]
        if right_shell_nodes:
            uw = [unwrap(ang[n], a0) for n in right_shell_nodes]
            a_min, a_max = min(uw), max(uw)
            a_min_p = max(a0, a_min - pad); a_max_p = min(a0 + 2*math.pi, a_max + pad)
            ax2.add_patch(Arc((0,0), 2*radius, 2*radius, angle=np.degrees(a0),
                             theta1=np.degrees(a_min_p - a0), theta2=np.degrees(a_max_p - a0),
                             linewidth=arc_lw, alpha=arc_alpha, linestyle='--', edgecolor=right_cmap(s)))
            amid = 0.5*(a_min_p + a_max_p)
            ax2.text((1.2*radius+0.06*dr)*math.cos(amid+0.3)-0.5, (1.2*radius+0.06*dr)*math.sin(amid+0.3)-0.7, r"$S_{2,"+str(s)+"}$",
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))
    ax2.text(-0.2, 0.2, r"$\beta$", ha='center', va='bottom',
           fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
    # title + push content upward
    #ax.set_title(title, fontsize=11, pad=4)
    R1 = (max_shell2 + 0.9) * dr
    R2 = (max_shell + 0.8) * dr
    ax1.text(-0.1,0.9,'a)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')

    ax2.text(1.06,0.9,'b)', transform=ax1.transAxes, fontsize=10, fontstyle='oblique')
    ax2.set_title("Nonsymmetric states", fontsize=10,y=1.045)
    ax2.set_xlim(-R1, R1)
    ax2.set_ylim(-R1, 0.75*R1)  # trims unused top space
    ax2.axis('off')
    

    # Tidy up frame and limits
    
    ax1.set_xlim(-R2, R2)
    ax1.set_ylim(-R2, R2)
    ax1.axis('off')
    ax1.set_title("Shells of a Cayley tree", fontsize=10,y=0.98)
    #plt.tight_layout()
    #plt.show()
    filename="cayley_shells.pdf"
    plt.savefig(filename)
    plt.close()
    

def main():
    #plot_DoS_phasediag()
    #plot_phases()
    #example_states_mu0()
    #example_states_different_mu()
    #plot_phases_nonzero_mu()
    #plot_Cayley_phasediag()
    #plot_phase_uniform()
    combined_figure_shells()


    
main()
