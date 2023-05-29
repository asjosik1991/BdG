import numpy as np
import BdG_on_graph as bdg
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pickle
import test_functions as test

def main():

    t=1

    T_array=np.array([0.005,0.2])
    V_array=np.array([1.5])
    mu_array=np.array([0])  
    fractal_iter=0
    alpha=0
    
    size=21
    mode="square"
    
    lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=True, noise=True)
    bdg.calculate_diagram(lattice_sample, V_array, mu_array, T_array)

    d=bdg.load_diagram(lattice_sample)
    Deltas=d['Deltas']
    rhos={}
    del_arrays={}
    for x in Deltas:
        print(x)
        V=x[0]
        T=x[1]
        mu=x[2]
        lattice_sample = bdg.Lattice(t, mode, size, alpha=alpha, fractal_iter=fractal_iter, pbc=True, noise=True)
        bdg_sample=bdg.BdG(lattice_sample,V,T,mu, Delta=Deltas[x])
        rho=bdg_sample.local_stiffness(0.001)
        rhos[x]=rho
        print("mean rho", np.mean(rho))

        
    diagram={'lattice_sample':lattice_sample, 'V':d['V'], 'mu':d['mu'], 'T':d['T'], 'rhos':rhos, 'dels':del_arrays}
    filename="rhos_mode={}_size={}_fractiter={}_delholes={}.pickle".format(lattice_sample.mode, lattice_sample.size,lattice_sample.fractal_iter, round(lattice_sample.alpha,2))
    pickle.dump(diagram, file = open(filename, "wb"))
    
    drhos=bdg.load_diagram(lattice_sample,suffix="rhos")
    rhos=drhos['rhos']
    #print(rhos)
    
    for x in Deltas:
        print(x)
        V=x[0]
        T=x[1]
        mu=x[2]

        bdg_sample=bdg.BdG(lattice_sample,V,T,mu, Delta=Deltas[x])
           
        #title=' V={} T={} $\mu$={}_mode={}_fractiter={}_delholes={}'.format(V,T,mu, mode, fractal_iter, alpha)
        title=''
        bdg_sample.field_plot(bdg_sample.Delta, fieldname='test_Delta',title='$\Delta$'+title)
        bdg_sample.field_plot(np.abs(np.real(rhos[x])), fieldname='test_rho',title='$\\rho$'+title)



         
main()
