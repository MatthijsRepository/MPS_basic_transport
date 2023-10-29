import os
os.chdir("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_basic_transport")

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt




"""
Plan
---------
Initialize a perfect half-state
Initialize a system Hamiltonian, simple XY with a magnetic field in z direction

Create procedure to make an MPO of the Hamiltonian
Create application procedure
Create re-truncation procedure?

Create SVD procedure?
Create canonicalization procedure

Renormalization procedure?

Expectation value calculator
Plot function of expval
"""



class MPS:
    def __init__(self, N, d, chi):
        self.N = N
        self.d = d
        self.chi = chi
        
        self.A_mat = np.zeros((N,chi,chi,d), dtype=complex) 
        #self.B_mat = np.zeros((N,d,chi,chi), dtype=complex)
        self.sup_A_mat = np.zeros((N, chi**2, chi**2, d**2), dtype=complex)
        
        self.Lambda_mat = np.zeros((N+1,chi),dtype=complex)
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)
        self.canonical_site = None
        return


    def initialize_halfstate(self):
        """ Initializes the MPS into a product state of |x+> eigenstates """
        #self.B_mat[:,:,0,0] = 1/np.sqrt(2)
        self.A_mat[:,0,0,:] = 1/np.sqrt(2)
        self.Lambda_mat[:,0] = 1
        self.Gamma_mat[:,:,0,0] = 1/np.sqrt(2)
        
        #self.locsize[:,:] = 1
        arr = np.arange(0,self.N+1)
        arr = np.minimum(arr, self.N-arr)
        arr = np.minimum(arr,self.chi)               # For large L, d**arr returns negative values, this line prohibits this effect
        self.locsize = np.minimum(self.d**arr, self.chi)
        return
    
    def initialize_flipstate(self):
        """ Initializes the MPS into a product of alternating up/down states """
        self.Lambda_mat[:,0] = 1
        for i in range(self.N):
            if (i%2==0):
                self.A_mat[i,0,0,0] = 1
                self.Gamma_mat[i,0,0,0] = 1
            else:
                self.A_mat[i,0,0,1] = 1
                self.Gamma_mat[i,1,0,0] = 1
               
        #self.locsize[:,:] = 1
        arr = np.arange(0,self.N+1)
        arr = np.minimum(arr, self.N-arr)
        arr = np.minimum(arr,self.chi)               # For large L, d**arr returns negative values, this line prohibits this effect
        self.locsize = np.minimum(self.d**arr, self.chi)
        return 
    
    def initialize_upstate(self):
        """ Initializes the MPS into a product state of up states """
        self.A_mat[:,0,0,0] = 1
        self.Lambda_mat[:,0] = 1
        self.Gamma_mat[:,0,0,0] = 1
        
        arr = np.arange(0,self.N+1)
        arr = np.minimum(arr, self.N-arr)
        arr = np.minimum(arr,self.chi)               # For large L, d**arr returns negative values, this line prohibits this effect
        self.locsize = np.minimum(self.d**arr, self.chi)
        return
    
    def give_A(self):
        return self.A_mat
    
    def give_LG(self):
        return self.Lambda_mat, self.Gamma_mat
    
    def give_locsize(self):
        return self.locsize
    
    
    def apply_twosite_new(self, TimeOp, i):
        normalize=True
        
        theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1))  #(chi, chi, d) -> (chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi, d, chi) 
        theta = np.tensordot(theta, self.Gamma_mat[i+1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+2,:]), axes=(3,0)) #(chi, d, d, chi)
        theta_prime = np.tensordot(theta,TimeOp[i,:,:,:,:],axes=([1,2],[2,3])) #(chi,chi,d,d)              # Two-site operator
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d*self.chi, self.d*self.chi)) #first to (d, chi, d, chi), then (d*chi, d*chi) # danger!
        #Singular value decomposition
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
        #truncation
        if normalize:
            self.Lambda_mat[i+1,:] = Y[:self.chi]*1/np.linalg.norm(Y[:self.chi])
        else:
            self.Lambda_mat[i+1,:] = Y[:self.chi]
        
        X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))  # danger!
        inv_lambdas = np.ones(self.locsize[i], dtype=complex)
        inv_lambdas *= self.Lambda_mat[i, :self.locsize[i]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i],:self.locsize[i+1]],axes=(1,1)) #(chi, d, chi)
        self.Gamma_mat[i, :, :self.locsize[i],:self.locsize[i+1]] = np.transpose(tmp_gamma,(1,0,2))
        
        Z = np.reshape(Z[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = np.ones(self.locsize[i+2], dtype=complex)
        inv_lambdas *= self.Lambda_mat[i+2, :self.locsize[i+2]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(Z[:,:self.locsize[i+1],:self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
        self.Gamma_mat[i+1, :, :self.locsize[i+1],:self.locsize[i+2]] = np.transpose(tmp_gamma,(0, 1, 2))    
        return 
     
    
    def TEBD_purestate(self, TimeOp):
        """
        for i in range(0, self.N-1):
            self.apply_twosite(TimeOp, i)
        """
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp, i)
            #self.apply_twosite_less_fancy(TimeOp, i)
        for i in range(1, self.N-1, 2):
            self.apply_twosite(TimeOp, i)
            #self.apply_twosite_less_fancy(TimeOp, i)
        
        #norm = self.calculate_vidal_inner_product(self)
        #self.Lambda_mat[1:] = self.Lambda_mat[1:]/(norm**(1/2*self.N))
        #"""
        return
    
    
    def construct_superket(self):
        """ Constructs a superket of the density operator, following D. Jaschke et al. (2018) """
        A_hermitian = np.conj(self.A_mat)
        for i in range(self.N):
            self.sup_A_mat[i,:,:,:] = np.tensordot(self.A_mat[i], A_hermitian[i], axes=0)
        return
  
    def apply_MPO_locsize(self, MPO_L, MPO_M, MPO_R):
        #MPO's must be in shape (d, d', 1, w1, w2) for kronecker product and einsum to work
        MPO_size = np.shape(MPO_L)[-1]        
        
        A_L = self.A_mat[0, :, :self.locsize[0], :self.locsize[1]]
        A_L = np.kron(MPO_L, A_L)
        #self.locsize[0,:] = np.multiply(self.locsize[0], np.array([1,MPO_size])) 
        self.A_mat[0,:, :self.locsize[0], :self.locsize[1]] = np.einsum('aiibc', A_L)
        
        for i in range(1, self.N-1):
            A_i = self.A_mat[i,:, :self.locsize[i], :self.locsize[i+1]]
            A_i = np.kron(MPO_M, A_i)
            #self.locsize[i,:] *= MPO_size 
            self.A_mat[i,:, :self.locsize[i], :self.locsize[i+1]] = np.einsum('aiibc', A_i)
            
        A_R = self.A_mat[self.N-1,:, :self.locsize[self.N-1], :self.locsize[self.N]]
        A_R = np.kron(MPO_R, A_R)
        #self.locsize[self.N-1] = np.multiply(self.locsize[self.N-1], np.array([MPO_size,1]))
        self.A_mat[self.N-1,:, :self.locsize[self.N-1], :self.locsize[self.N]] = np.einsum('aiibc', A_R)
        #self.locsize *= MPO_size
        return
    
    def expval(self, Op, singlesite, site):
        
        if singlesite:
            #theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,0)) #(chi, chi, d)
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            #theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(1,0))    #(chi,d,chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            #theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            #result = np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2],[0,2,1]))
            result = np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2],[0,2,1]))
            return np.real(result)
 
        result = 0      #calculate expval for entire chain
        for i in range(self.N):
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d) 
            result += np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2],[0,2,1]))
        return np.real(result)/self.N
    
    def calculate_vidal_norm(self):
        m_total = np.eye(chi)
        for j in range(0, self.N):        
            #sub_tensor = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(1,0)) #(chi, d, chi)
            sub_tensor = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            #mp = np.tensordot(np.conj(sub_tensor),sub_tensor,axes = (1,1)) #(chi, chi, chi, chi)
            mp = np.tensordot(np.conj(sub_tensor), sub_tensor, axes=(0,0)) #(chi, chi, chi, chi)
            #m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))     
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
        return np.real(m_total[0,0])

"""    
    def calculate_vidal_inner_product(self, MPS2):
        m_total = np.eye(self.chi)
        temp_lambdas, temp_gammas = MPS2.give_LG()
        for i in range(0,self.N):
            st1 = np.tensordot(self.Gamma_mat[i,:,:,:], np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[i,:,:,:], np.diag(temp_lambdas[i+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st1), st2, axes=(0,0))
            m_total = np.tensordot(m_total, mp, axes=([0,1],[0,2]))
        return abs(m_total[0,0])
"""   
########################################################################################  



def calculate_inner_product(A1, A2):
    N = np.shape(A1)[0]
    chi_1 = np.shape(A1)[-1]
    chi_2 = np.shape(A2)[-1]
    if chi_1 != chi_2:
        print("Failed to calculate inner product, different bond dimensions")
        return
    temp = np.eye(chi_1)
    for i in range(N):
        temp = np.einsum('aib, ic -> abc', np.conj(A1[i]), temp)    #since since hermitean conjugate requires transpose
        temp = np.einsum('ibj, idj -> bd', temp, A2[i])
        #temp = np.tensordot(np.conj(A1[i]), temp, axes=(1,1))      #here first axis is 1 instead of 2 due to the transpose
        #temp = np.tensordot(temp, A2[i], axes=([0,2],[0,1]))
    return abs(temp[0,0])


def Create_Ham(h, J, N, d):
    #creates XY model Hamiltonian with magnetic field h in z direction
    
    SX = np.kron(Sx, Sx)
    SY = np.kron(Sy, Sy)
    SZ_L = np.kron(Sz, np.eye(2))
    SZ_R = np.kron(np.eye(2), Sz)
    SZ_M = (SZ_L + SZ_R)
    
    H_L = h*(SZ_L + SZ_R/2) + J*(SX + SY)
    H_R = h*(SZ_L/2 + SZ_R) + J*(SX + SY)
    H_M = h*SZ_M/2 + J*(SX + SY)
    
    
    H_arr = np.zeros((N-1, d**2, d**2), dtype=complex)
    
    for i in range(1,N-2):
        H_arr[i,:,:] = H_M
    H_arr[0,:,:] = H_L
    H_arr[N-2,:,:] = H_R
    return H_arr

def Create_TimeOp(H, delta, N, d):
    #H = np.reshape(H, (N-1, d**2, d**2))
    U = np.ones((N-1, d**2, d**2), dtype=complex)
    
    """
    Hcn0 = create_crank_nicolson(H[0], delta, N, d)
    HcnM = create_crank_nicolson(H[1], delta, N, d)
    HcnN = create_crank_nicolson(H[N-2], delta, N, d)
    
    U[0,:,:] = Hcn0
    U[N-2,:,:] = HcnN
    U[1:N-2,:,:] *= HcnM #H from sites 2 and 3 - we use broadcasting
    U = np.around(U, decimals=15)        #Rounding out very low decimals 
    """
    
    U[0,:,:] = expm(-1j*delta*H[0])
    U[N-2,:,:] = expm(-1j*delta*H[N-2])
    U[1:N-2,:,:] *= expm(-1j*delta*H[1]) #H from sites 2 and 3 - we use broadcasting
    #"""
    
    U = np.around(U, decimals=15)        #Rounding out very low decimals 
    return np.reshape(U, (N-1,d,d,d,d)) 


def create_crank_nicolson(H, dt, N, d):
    H_top=np.eye(H.shape[0])-1j*dt*H/2
    H_bot=np.eye(H.shape[0])+1j*dt*H/2
    return np.linalg.inv(H_bot).dot(H_top)

def Create_Ham_MPO(J, h):
    # J*Szi*Szi+1 + h*Sz
    H_L = np.array([[-h*Sz], [Sz], [np.eye(2)]]).transpose((2,3,1,0)).reshape((2,2,1,1,3)) #transpose((1,0,2,3))
    H_M = np.array([[np.eye(2), np.zeros((2,2)), np.zeros((2,2))],[Sz, np.zeros((2,2)), np.zeros((2,2))],[-h*Sz, J*Sz, np.eye(2)]]).transpose((2,3,0,1)).reshape((2,2,1,3,3))
    H_R = np.array([np.eye(2), J*Sz, -h*Sz]).transpose((1,2,0)).reshape((2,2,1,3,1))
    return H_L, H_M, H_R



####################################################################################


N=5
d=2
chi=10
steps = 700
dt = 0.01

h=-2
J=-0.1

Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

####################################################################################






MPS1 = MPS(N,d,chi)
#MPS1.initialize_halfstate()
MPS1.initialize_flipstate()



Ham = Create_Ham(h, J, N, d)


"""
IMPORTANT NOTE: the additional reshaping steps are indeed redundant
"""

"""
H_arr = np.ones((N-1,d,d,d,d), dtype=complex)
H_arr[0,:,:,:] = np.reshape(Ham[0],(d,d,d,d))
H_arr[N-2,:,:,:] = np.reshape(Ham[N-2], (d,d,d,d))
H_arr[1:N-2,:,:,:] *= np.reshape(Ham[1], (d,d,d,d))

O_arr = np.zeros((N-1,d,d,d,d), dtype=complex)

def operator_Create(H,dt,d): 
    H_top=np.eye(H.shape[0])-1j*dt*H/2
    H_bot=np.eye(H.shape[0])+1j*dt*H/2
    Hop = np.linalg.inv(H_bot).dot(H_top)
    return np.reshape(Hop,(d,d,d,d)) 

for i in range(0,N-1):
    O_arr[i,:,:,:,:] = operator_Create(np.reshape(H_arr[i,:,:,:,:],(d**2,d**2)),dt,d)    

TimeOp = O_arr

#print(O_arr[1])
#print(operator_Create(Ham[1], dt, d))
"""






TimeOp = Create_TimeOp(Ham, dt, N, d)


norm = np.zeros(steps)
exp_sz = np.zeros((N,steps))

for t in range(steps):
    for i in range(0, N-1, 2):
        MPS1.apply_twosite_new(TimeOp, i)
    for i in range(1, N-1, 2):
        MPS1.apply_twosite_new(TimeOp, i)
        
    norm[t] = MPS1.calculate_vidal_norm()
    #exp_sz[t] = MPS1.expval(Sz, False, 0)
    for j in range(N):
        exp_sz[j,t] = MPS1.expval(Sz, True, j)
    
plt.plot(norm)
plt.show()
for j in range(N):
    plt.plot(exp_sz[j,:])
plt.show()








#a = MPS1.calculate_vidal_inner_product(MPS2)
#b = MPS1.expval(Sz, False, 0)

#print(b)
#print()
#print(Ham[1])
#print(Ham[0])
#print(TimeOp[1])
#print(TimeOp[0])

#lambdas, gammas = MPS1.give_LG()
#print("gammas")
#print(gammas[1,0])
#print(gammas[1,1])
#print()

#MPS1_inner = np.zeros(steps)
#MPS2_inner = np.zeros(steps)
#exp_sz = np.zeros(steps)




"""
def main():
    for i in range(steps):
        #print(i)
        MPS1.TEBD_purestate(TimeOp)
        #MPS1.TEBD_purestate_verliezen(TimeOp)
        
        #lambdas, gammas = MPS1.give_LG()
        #print("gammas")
        #print(gammas[1,0])
        #print(gammas[1,1])
        

        exp_sz[i] = MPS1.expval(Sz, False, 0)
        MPS1_inner[i] = MPS1.calculate_vidal_norm() #MPS1.calculate_vidal_inner_product(MPS1)
        
        #if MPS1_inner[i]<0.98:
        #    print(gammas[2,0])
        #    print(gammas[2,1])
        
        if (i%1==0 or i==0):
            print()
            print(i)
            print("<Sz> total, normalization")
            print(exp_sz[i])
            print(MPS1_inner[i])
            print("<Sz> site by site")
            #for j in range(0,N):
                #print(np.round(MPS1.expval(Sz, True, j), decimals=4))
        
    #print(np.round(gammas[1], decimals=4))
    plt.plot(MPS1_inner, color="red", label="norm")
    plt.plot(exp_sz, color="blue", label="<Sz>")
    plt.grid()
    plt.legend()
    plt.show()
    return
        
 
main()  

#MPS1.apply_MPO_locsize(H_L, H_M, H_R)

#a = MPS1.expval(Sz, True, 0)
#print(a)

"""





"""
a = calculate_inner_product(MPS1.give_A(), MPS2.give_A())
print(a) 
print(MPS1.give_A())

MPS1.apply_MPO_locsize(H_L, H_M, H_R)
a = calculate_inner_product(MPS1.give_A(), MPS2.give_A())
print(a)   
print(MPS1.give_A())
 """   
    







# =============================================================================
#     def get_coefficient(self, d_array):
#         self.Lambda = np.diag(self.Lambda_mat)
#         self.Gamma = 0
#         return
# =============================================================================
        
        






























