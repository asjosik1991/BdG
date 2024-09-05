import numpy as np
from scipy import integrate


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

def BCS_gap(dos, V,T):
    
    def gap_kernel(Delta):
        def func(u):
            return Delta*V*dos(u)*np.tanh(np.sqrt(u**2+Delta**2)/(2*T))/(2*np.sqrt(u**2+Delta**2))
        return func
          
    def gap_integral(Delta):
        return integrate(gap_kernel(Delta),-10,10)
       
    print("Calculation of BCS gap")

    Delta=1
    step=0
    V=0.5*V

    while True:
        Delta_1=gap_integral(Delta)
        Delta_2=gap_integral(Delta_1)
        Delta_next= Delta-(Delta_1-Delta)**2/(Delta_2-2*Delta_1+Delta)
        error=np.abs(Delta-Delta_next)
        Delta=Delta_next
        print("step", step, "error", error, "Delta", Delta)
        step += 1
        if error<10**(-6):
            break
    return Delta

"Test functions for homogeneous flat case"
#Fermi function
def F(epsilon,T):
    return 1/(np.exp(epsilon/T)+1)

#analytical formula for uniform case, for tests
def uniform_2D_correlation_function(size, T, Delta=0, state='normal'):
    
    N_qy=100
    k=np.linspace(0, 2*np.pi*(1-1/size), size)
    q_y=np.linspace(2*np.pi/size, 2*np.pi*(1-1/size), N_qy)
    Lambda=np.zeros(N_qy)
    
    if state=='normal':
        
        for i in range(N_qy):
            for k_x in k:
                for k_y in k:
                    eps=-2*(np.cos(k_x)+np.cos(k_y))
                    eps_qy=-2*(np.cos(k_x)+np.cos(k_y+q_y[i]))
                    Lambda[i]+=np.real(-8/(size**2)*(np.sin(k_x)**2)*(F(eps,T)-F(eps_qy,T))/(eps-eps_qy+1j*10**(-6)))
            
        #kinetic energy from limit
        K_simple=0
        K_withderivative=0
        for k_x in k:
            for k_y in k:
                
                eps=-2*(np.cos(k_x)+np.cos(k_y))
                K_simple+=4*np.cos(k_x)*F(eps,T)/(size**2)
                K_withderivative+=8*(np.sin(k_x)**2)/(4*T*(size**2)*(np.cosh(eps/(2*T))**2))

        print("analytic kinetic energy", K_simple, "\n analytical limit", K_withderivative, "\n numerical limit", Lambda[0])
        
        return q_y, Lambda, K_simple
    
    if state=='super':
        
        for i in range(N_qy):
            for k_x in k:
                for k_y in k:
                    eps=-2*(np.cos(k_x)+np.cos(k_y))
                    eps_qy=-2*(np.cos(k_x)+np.cos(k_y+q_y[i]))
                    E=np.sqrt(eps**2 + Delta**2)
                    E_qy=np.sqrt(eps_qy**2 + Delta**2)
                    L=0.5*(1 + (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    P=0.5*(1- (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    Lambda[i]+=4/(size**2)*(np.sin(k_x)**2)*(L*(1/(1j*10**(-6)+E-E_qy)+1/(-1j*10**(-6)+E-E_qy)*(F(E,T)-F(E_qy,T)))
                                                             +P*(1/(1j*10**(-6)+E+E_qy)+1/(-1j*10**(-6)+E+E_qy)*(1-F(E,T)-F(E_qy,T))))
                    
                    E=-np.sqrt(eps**2 + Delta**2)
                    E_qy=-np.sqrt(eps_qy**2 + Delta**2)
                    L=0.5*(1 + (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    P=0.5*(1- (eps*eps_qy+Delta**2)/(E*E_qy+10**(-6)))
                    Lambda[i]+=4/(size**2)*(np.sin(k_x)**2)*(L*(1/(1j*10**(-6)+E-E_qy)+1/(-1j*10**(-6)+E-E_qy)*(F(E,T)-F(E_qy,T)))
                                                             +P*(1/(1j*10**(-6)+E+E_qy)+1/(-1j*10**(-6)+E+E_qy)*(1-F(E,T)-F(E_qy,T))))
             
            return q_y, Lambda

def Aitken_uniform_BdG(size, V,T,mu): #Aitken delta-squired process to increase convergence rate
    
    def gap_integral(x, energies_sq, V,size):
        E=np.sqrt(energies_sq+x**2)
        return np.sum(V*x*(np.ones(size**2)-2*F(E,T))/E)
       
    print("Calculation of uniform periodic square BdG with Aitken method")
    k_array=np.linspace(0, 2*np.pi*(1-1/size), size)
    energies_x=-2*np.cos(k_array)
    energies_y=np.copy(energies_x)-mu*np.ones(size)
    energies=np.transpose([np.tile(energies_x, size), np.repeat(energies_y, size)])
    energies=np.sum(energies,axis=-1)
    energies_sq=energies**2
    Delta=1
    step=0
    V=0.5*V/size**2

    while True:
        Delta_1=gap_integral(Delta, energies_sq,V,size)
        Delta_2=gap_integral(Delta_1, energies_sq,V,size)
        Delta_next= Delta-(Delta_1-Delta)**2/(Delta_2-2*Delta_1+Delta)
        error=np.abs(Delta-Delta_next)
        Delta=Delta_next
        print("step", step, "error", error, "Delta", Delta)
        step += 1
        if error<10**(-6):
            break
    return Delta

def uniform_2D_BdG(size,V,T,mu,mode="square"):
    print("Calculation of uniform periodic BdG")
    k_array=np.linspace(0, 2*np.pi*(1-1/size), size)
    if mode=="square":
        energies_x=-2*np.cos(k_array)
        energies_y=np.copy(energies_x)-mu*np.ones(size)
        energies=np.transpose([np.tile(energies_x, size), np.repeat(energies_y, size)])
        energies=np.sum(energies,axis=-1)
    if mode=="triangle":
        energies=np.zeros(size**2)
        i=0
        for k_x in k_array:
            for k_y in k_array:
                energies[i]=-2*(np.cos(k_x)+np.cos(k_y)+np.cos(k_x+k_y))-mu
                i+=1
 
    energies_sq=energies**2
    Delta=1
    step=0
    V=0.5*V/size**2
    while True:
           E=np.sqrt(energies_sq+Delta**2)
           aux_array=V*Delta*(np.ones(size**2)-2*F(E,T))/E
           Delta_next=np.sum(aux_array)
           error=np.abs(Delta-Delta_next)
           Delta=Delta_next
           print("step", step, "error", error, "Delta", Delta)
           step += 1
           if error<10**(-6):
               break
           
    return Delta
