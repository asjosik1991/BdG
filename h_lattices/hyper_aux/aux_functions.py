import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
from matplotlib import ticker


def bethe_dos3(s):
    return bethe_dos(2,s)

def hyper_dos83(s):
    bs=[2.1092519737,2.1117327570,2.0929965018,2.1343590649,2.1147900908,2.0409742217,996022307/451765525, 5117344/2432325, 53675/27552,1435/608,192/95,19/10,5/2,2,2,3]
    return aprx_hyp_dos(bs, s)

def bethe_dos(q,s):
    if abs(s)<2*np.sqrt(q):
        #print("check", abs(s), 2*np.sqrt(q))
        return (q+1)/(2*np.pi)*np.sqrt(4*q-s**2)/((q+1)**2-s**2)
    else:
        #print("check2", abs(s), 2*np.sqrt(q))
        return 0

def aprx_hyp_dos(bs, s, eps=10**(-4)): #data can be taken from arXiv:2304.02382
    
    z=s-1j*eps
    if np.imag(np.sqrt(z**2-4*bs[0]))<0:
        G=(z-np.sqrt(z**2-4*bs[0]))/(2*bs[0])
    else:
        G=(z+np.sqrt(z**2-4*bs[0]))/(2*bs[0])
    for i in range(len(bs)-1):
        G=1/(z-bs[i+1]*G)
    return np.imag(G)/np.pi

def BCS_gap(dos, mu, T, V=1, Delta_seed=1):
    
    def gap_kernel(Delta):
        def func(u):
            #print("gap kernel test", Delta, V, mu, T, u, dos(u-mu))
            return Delta*V*dos(u)*np.tanh(np.sqrt((u-mu)**2+Delta**2)/(2*T))/(2*np.sqrt((u-mu)**2+Delta**2))
        return func
          
    def gap_integral(Delta):
        kernel=gap_kernel(Delta)
        return integrate.quad(kernel,-4,4)[0]
       
    print("Calculation of BCS gap mu=", mu, "T=", T)

    Delta=Delta_seed
    step=0

    while True:
        Delta_1=gap_integral(Delta)
        Delta_2=gap_integral(Delta_1)
        if np.abs(Delta_2-Delta_1)<10**(-6):
            Delta=Delta_2
            print("step", step, "error", np.abs(Delta_2-Delta_1), "Delta", Delta)
            break
        Delta_next= Delta-(Delta_1-Delta)**2/(Delta_2-2*Delta_1+Delta)
        error=np.abs(Delta-Delta_next)
        Delta=Delta_next
        print("step", step, "error", error, "Delta", Delta)
        step += 1
        if error<10**(-6) or Delta<10**(-6):
            break
    return Delta

def Delta_muslice(T_array):
    mu=0
    V=1
    dos=bethe_dos3
    Deltas=[]
    Delta_seed=1
    for T in T_array:
        print("calculating V=",V,"mu=",mu,"T=",T)
        if np.max(Delta_seed)<10**(-6):
            Deltas.append(0)
            continue
        Delta=BCS_gap(dos, mu, T,V, Delta_seed)
        Deltas.append(Delta)
        Delta_seed=Delta
    
    return Deltas
    

def plot_DoS(dos, energy_range):
    dos_array=[]
    for s in energy_range:
        dos_array.append(dos(s))
    plt.rc('font', family = 'serif', serif = 'cmr10')
    rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(9.6,7.2))
    ax.locator_params(nbins=5)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(r'$\rho(s)$',fontsize=26)
    plt.xlabel(r's',fontsize=26)
    plt.plot(energy_range, dos_array)

    plt.title("\{8,3\} lattice DoS", fontsize=32)
    #plt.title("Bethe lattice DoS", fontsize=20)
    #plt.show()
    plt.savefig("dos.pdf")
    plt.close()
    
#create general array of Delta depending on different parameters for a given system
def calculate_exactdiagram(dos, V_array, mu_array, T_array):
    Deltas={}
    for V in V_array:
        for mu in mu_array:
            Delta_seed=1
            for T in T_array:
                print("calculating V=",V,"mu=",mu,"T=",T)
                if np.max(Delta_seed)<10**(-6):
                    Deltas[(V,T,mu)]=0
                    continue
                
                Deltas[(V,T,mu)]=BCS_gap(dos, mu, T,V, Delta_seed)
                Delta_seed=Deltas[(V,T,mu)]


    diagram={'V':V_array, 'mu':mu_array, 'T':T_array, 'Deltas':Deltas}
    filename="diagram_hyperbolic.pickle"
    pickle.dump(diagram, file = open(filename, "wb"))

def load_exactdiagram():  
    filename="diagram_hyperbolic.pickle"
    try:
        diagram=pickle.load(file = open(filename, "rb"))
        return diagram
    except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
        return -1

def plot_exactdiagram(diagram):
    
    def plotting(field, legend):
        
        
        fig, ax = plt.subplots(figsize=(9.6,7.2))
        plt.rc('font', family = 'serif', serif = 'cmr10')
        rc('text', usetex=True)
        plt.xlabel(r'$\mu$',fontsize=26)
        plt.ylabel(r'$T$',fontsize=26)
        ax.locator_params(nbins=5)
        #plt.title("Bethe lattice",fontsize=32)
        plt.title(r'$\{8,3\}$ lattice',fontsize=32)


        plt.imshow(field, vmin=field.min(), vmax=field.max(), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect = np.abs((x.max() - x.min())/(y.max() - y.min())))
        cbar=plt.colorbar()
        cbar.set_label(legend, fontsize=26, rotation=0, labelpad=-35, y=1.1)
        cbar.ax.tick_params(labelsize=24)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        filename="diagram_hyperbolic.pdf"

        plt.savefig(filename)
        plt.close()
        #plt.show()


    if len(diagram['V'])==1:
        V=diagram['V'][0]
        x=diagram['mu']
        y=diagram['T']
        Deltas=diagram['Deltas']
        print(x)
        print(y)
        print(Deltas)
        Delta=np.zeros((len(y),len(x)))      

        for i in range(len(y)):
            for j in range(len(x)):
                Delta[i,j]=Deltas[(V,y[i],x[j])]
 
        plotting(Delta,r'$\Delta$')
