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
        
        self.A_mat = np.zeros((N,d,chi,chi), dtype=complex)
        #self.B_mat = np.zeros((N,d,chi,chi), dtype=complex)
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
        self.sup_A_mat = np.zeros((N,d**2, chi**2, chi**2), dtype=complex)
        
        self.locsize = np.zeros(N+1, dtype=int)
        self.canonical_site = None
        return


    def initialize_halfstate(self):
        """ Initializes the MPS into a product state of |x+> eigenstates """
        #self.B_mat[:,:,0,0] = 1/np.sqrt(2)
        self.A_mat[:,:,0,0] = 1/np.sqrt(2)
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
                self.A_mat[i,1,0,0] = 1
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
    
    def apply_twosite(self, TimeOp, i):
        theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi, d, chi)
        theta = np.tensordot(theta, self.Gamma_mat[i+1,:,:,:],axes=(2,1)) #(chi, d, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+2,:]), axes=(3,0)) #(chi, d, d, chi)
        theta_prime = np.tensordot(theta,TimeOp[i,:,:,:,:],axes=([1,2],[2,3]))  #(chi, chi, d, d)
        #print(i)
        #print(theta_prime[:,:,0,0])
        #print(theta_prime[:,:,0,1])
        #print(theta_prime[:,:,1,0])
        #print(theta_prime[:,:,1,1])
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d*self.chi, self.d*self.chi)) # danger!
        
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
        
        #inv_lambdas are part of the problem due to numerical errors, so low values in Y are rounded to 0
        Y[Y < 10**-10] = 0
        self.Lambda_mat[i+1,:] = Y[:chi]/np.linalg.norm(Y[:chi])
        #self.Lambda_mat[i+1, self.Lambda_mat[i+1]<10**-6] = 0        
        #if (len(Y[Y>10]) >0):
        #    print("High lambdas encountered")
        #    print(Y[:chi])
        

        X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))  # danger!         
        inv_lambdas = np.ones(self.locsize[i])
        inv_lambdas *= self.Lambda_mat[i, :self.locsize[i]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas), X[:, :self.locsize[i], :self.locsize[i+1]], axes=(1,1))
        tmp_gamma = tmp_gamma.transpose(1,0,2) # to ensure shape (d, chi, chi)
        self.Gamma_mat[i, :, :self.locsize[i], :self.locsize[i+1]] = tmp_gamma
    
        Z = np.reshape(Z[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))  # danger!
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = np.ones(self.locsize[i+2])
        inv_lambdas *= self.Lambda_mat[i, :self.locsize[i+2]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(Z[:, :self.locsize[i+1], :self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0))
        self.Gamma_mat[i+1, :, :self.locsize[i+1], :self.locsize[i+2]] = tmp_gamma 
        return
    
    def TEBD_purestate(self, TimeOp):
        """
        for i in range(0, self.N-1):
            self.apply_twosite(TimeOp, i)
        """
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp, i)
        for i in range(1, self.N-1, 2):
            self.apply_twosite(TimeOp, i)
        
        #norm = self.calculate_vidal_inner_product(self)
        #self.Lambda_mat[1:] = self.Lambda_mat[1:]/(norm**(1/2*self.N))
        #"""
        return
    
    def TEBD_purestate_old(self, TimeOp):
        """ Performs a single TEBD sweep over the Lambdas and the Gammas, code is almost identical to that used in the BEP """
        for i in range(0,self.N-1):
            theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi, d, chi)
            theta = np.tensordot(theta, self.Gamma_mat[i+1,:,:,:],axes=(2,1)) #(chi, d, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+2,:]), axes=(3,0)) #(chi, d, d, chi)
            theta_prime = np.tensordot(theta,TimeOp[i,:,:,:,:],axes=([1,2],[2,3]))  #(chi, chi, d, d)
            theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d * self.chi,self.d * self.chi)) # danger!
            
            X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
            
            #inv_lambdas are part of the problem due to numerical errors, so low values in Y are rounded to 0
            Y[Y < 10**-6] = 0
            self.Lambda_mat[i+1,:] = Y[:chi]*1/np.linalg.norm(Y[:chi])
            
            X = np.reshape(X[:self.d*self.chi,:self.chi], (self.d, self.chi, self.chi))  # danger!         
            inv_lambdas = np.ones(self.locsize[i])
            inv_lambdas *= self.Lambda_mat[i, :self.locsize[i]]
            inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
            tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:, :self.locsize[i], :self.locsize[i+1]], axes=(1,1))
            tmp_gamma = tmp_gamma.transpose(1,0,2) # to ensure shape (d, size1, size2)
            self.Gamma_mat[i, :, :self.locsize[i], :self.locsize[i+1]] = tmp_gamma
            
            Z = np.reshape(Z[0:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))  # danger!
            Z = np.transpose(Z,(0,2,1))
            inv_lambdas = np.ones(self.locsize[i+2])
            inv_lambdas *= self.Lambda_mat[i, :self.locsize[i+2]]
            inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
            tmp_gamma = np.tensordot(Z[:, :self.locsize[i+1], :self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0))
            self.Gamma_mat[i+1, :, :self.locsize[i+1], :self.locsize[i+2]] = tmp_gamma 
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
        
        A_L = self.A_mat[0,:, :self.locsize[0], :self.locsize[1]]
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
        if singlesite:  #calculate expval for a single site
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta, np.diag(self.Lambda_mat[site+1,:]), axes=(2,0)) #(chi, d, chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            return np.real(np.tensordot(np.conj(theta_prime), theta, axes=([0,1,2],[0,2,1])))
 
        result = 0      #calculate expval for entire chain
        for i in range(self.N):
            theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta, np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(chi, d, chi)    
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            result += np.tensordot(np.conj(theta_prime), theta, axes=([0,1,2],[0,2,1]))
        return np.real(result)/self.N
    
    def calculate_vidal_norm(self):
        m_total = np.eye(self.chi)
        for i in range(0, self.N):
            st = np.tensordot(self.Gamma_mat[i,:,:,:], np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st), st, axes=(0,0))
            m_total = np.tensordot(m_total, mp, axes=([0,1],[0,2]))
        return abs(m_total[0,0])
    
    def calculate_vidal_inner_product(self, MPS2):
        m_total = np.eye(self.chi)
        temp_lambdas, temp_gammas = MPS2.give_LG()
        for i in range(0,self.N):
            st1 = np.tensordot(self.Gamma_mat[i,:,:,:], np.diag(self.Lambda_mat[i+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[i,:,:,:], np.diag(temp_lambdas[i+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st1), st2, axes=(0,0))
            m_total = np.tensordot(m_total, mp, axes=([0,1],[0,2]))
        return abs(m_total[0,0])
    
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
    
    """
    H_arr = np.zeros((N-1, d, d, d, d), dtype=complex)
     
    #reshape the (d*d)*(d*d) matrices into d*d*d*d matrices
    for i in range(1,N-2):      
        H_arr[i,:,:,:,:] = np.reshape(H_M, (d,d,d,d))
    H_arr[0,:,:,:,:] = np.reshape(H_L, (d,d,d,d))
    H_arr[N-2,:,:,:,:] = np.reshape(H_R, (d,d,d,d)) #N-2 since array length N-1 has last index N-2, and last operator acts on sites N-1 and N
    """
    return H_arr

def Create_TimeOp(H, delta, N, d):
    #H = np.reshape(H, (N-1, d**2, d**2))
    U = np.ones((N-1, d**2, d**2), dtype=complex)
    
    U[0,:,:] = expm(-1j*delta*H[0])
    U[N-2,:,:] = expm(-1j*delta*H[N-2])
    U[1:N-2,:,:] *= expm(-1j*delta*H[1]) #H from sites 2 and 3 - we use broadcasting
    U = np.around(U, decimals=15)        #Rounding out very low decimals 
    return np.reshape(U, (N-1,d,d,d,d)) 


def Create_Ham_MPO(J, h):
    # J*Szi*Szi+1 + h*Sz
    H_L = np.array([[-h*Sz], [Sz], [np.eye(2)]]).transpose((2,3,1,0)).reshape((2,2,1,1,3)) #transpose((1,0,2,3))
    H_M = np.array([[np.eye(2), np.zeros((2,2)), np.zeros((2,2))],[Sz, np.zeros((2,2)), np.zeros((2,2))],[-h*Sz, J*Sz, np.eye(2)]]).transpose((2,3,0,1)).reshape((2,2,1,3,3))
    H_R = np.array([np.eye(2), J*Sz, -h*Sz]).transpose((1,2,0)).reshape((2,2,1,3,1))
    return H_L, H_M, H_R



####################################################################################


N=5
d=2
chi=4
steps = 20
dt = 0.01

h=0
J=1
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

####################################################################################


MPS1 = MPS(N,d,chi)
MPS1.initialize_halfstate()
#MPS1.initialize_flipstate()

MPS2 = MPS(N,d,chi)
#MPS2.initialize_halfstate()
MPS2.initialize_flipstate()

#H_L, H_M, H_R = Create_Ham_MPO(1, 0)

Ham = Create_Ham(h, J, N, d)
TimeOp = Create_TimeOp(Ham, dt, N, d)
#print(TimeOp[1])


a = MPS1.calculate_vidal_inner_product(MPS2)
b = MPS1.expval(Sz, False, 0)
print(a)
print(b)
print()
#print(Ham[1])
#print(Ham[0])
#print(TimeOp[1])
#print(TimeOp[0])

lambdas, gammas = MPS1.give_LG()
#print("gammas")
#print(gammas[1,0])
#print(gammas[1,1])
print()

MPS1_inner = np.zeros(steps)
MPS2_inner = np.zeros(steps)
exp_sz = np.zeros(steps)


def main():
    for i in range(steps):
        #print(i)
        MPS1.TEBD_purestate(TimeOp)
        #MPS1.TEBD_purestate_verliezen(TimeOp)
        
        lambdas, gammas = MPS1.give_LG()
        #print("gammas")
        #print(gammas[1,0])
        #print(gammas[1,1])
        
        MPS2_inner[i] = MPS1.calculate_vidal_inner_product(MPS2)
        exp_sz[i] = MPS1.expval(Sz, False, 0)
        MPS1_inner[i] = MPS1.calculate_vidal_norm() #MPS1.calculate_vidal_inner_product(MPS1)
        
        #if MPS1_inner[i]<0.98:
        #    print(gammas[2,0])
        #    print(gammas[2,1])
        
        if (i%1==0 or i==0):
            print()
            print(i)
            print("inner initial, <Sz> total, normalization")
            print(MPS2_inner[i])
            print(exp_sz[i])
            print(MPS1_inner[i])
            print("<Sz> site by site")
            #for j in range(0,N):
                #print(np.round(MPS1.expval(Sz, True, j), decimals=4))
        
      
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
        
        




















