import numpy as np
import hyper_aux.aux_functions as haux
from matplotlib import rc

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
import effective_models.Cayley_tree as tree
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cmap

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

def main():
    #plot_DoS_phasediag()
    #plot_phases()
    #example_states_mu0()
    #example_states_different_mu()
    #plot_phases_nonzero_mu()
    #plot_Cayley_phasediag()
    #plot_phase_uniform()
    
main()
