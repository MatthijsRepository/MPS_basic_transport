import os
os.chdir("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_basic_transport")

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


class MPS:
    def __init__(self, N, d, chi):
        self.N = N
        self.d = d
        self.chi = chi
        
        self.A_mat = np.zeros((N,d,chi,chi), dtype=complex) 
        #self.B_mat = np.zeros((N,d,chi,chi), dtype=complex)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        self.canonical_site = None
        return

    def initialize_halfstate(self):
        """ Initializes the MPS into a product state of uniform eigenstates """
        #self.B_mat[:,:,0,0] = 1/np.sqrt(2)
        self.A_mat[:,:,0,0] = 1/np.sqrt(self.d)
        self.Lambda_mat[:,0] = 1
        self.Gamma_mat[:,:,0,0] = 1/np.sqrt(self.d)
        
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
                self.A_mat[i,self.d-1,0,0] = 1
                self.Gamma_mat[i,self.d-1,0,0] = 1
               
        #self.locsize[:,:] = 1
        arr = np.arange(0,self.N+1)
        arr = np.minimum(arr, self.N-arr)
        arr = np.minimum(arr,self.chi)               # For large L, d**arr returns negative values, this line prohibits this effect
        self.locsize = np.minimum(self.d**arr, self.chi)
        return 
    
    def initialize_up_or_down(self, up):
        """ Initializes the MPS into a product state of up or down states """
        if up:  #initialize each spin in up state
            i=0 
        else:   #initialize each spin in down state
            i=self.d-1
        self.A_mat[:,i,0,0] = 1
        self.Lambda_mat[:,0] = 1
        self.Gamma_mat[:,i,0,0] = 1
        
        arr = np.arange(0,self.N+1)
        arr = np.minimum(arr, self.N-arr)
        arr = np.minimum(arr,self.chi)               # For large L, d**arr returns negative values, this line prohibits this effect
        self.locsize = np.minimum(self.d**arr, self.chi)
        return
    
    def set_Gamma_Lambda(self, gammas, lambdas, locsize):
        """ Custom initialization of the MPS """
        self.Gamma_mat = gammas
        self.Lambda_mat = lambdas
        self.locsize = locsize
        return
    
    def set_Gamma_singlesite(self, site, matrix):
        """ sets a gamma matrices of a site to a desired matrix  """
        self.Gamma_mat[site,:,:,:] = matrix
        return   
    
    def give_A(self):
        """ Returns the 'A' matrices of the MPS """
        return self.A_mat
    
    def give_LG(self):
        """ Returns the Lambda and Gamma matrices of the MPS """
        return self.Lambda_mat, self.Gamma_mat
    
    def give_locsize(self):
        """ Returns the locsize variable """
        return self.locsize
    
    
    def construct_superket(self):
        """ Constructs a superket of the density operator, following D. Jaschke et al. (2018) """
        sup_A_mat = np.zeros((self.N, self.d**2, self.chi**2, self.chi**2), dtype=complex)
        for i in range(self.N):
            sup_A_mat[i,:,:,:] = np.kron(self.A_mat[i], np.conj(self.A_mat[i]))
        return sup_A_mat, self.locsize**2
    
    def construct_vidal_superket(self):
        """ Constructs a superket of the density operator in Vidal decomposition """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, self.chi**2, self.chi**2), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N, self.chi**2))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])
        sup_Lambda_mat[N,:] = np.kron(self.Lambda_mat[N], self.Lambda_mat[N])
        return sup_Gamma_mat, sup_Lambda_mat, self.locsize**2
    
   
    def apply_twosite(self, TimeOp, i, normalize):
        """ Applies a two-site operator to sites i and i+1 """
        #First the matrices lambda-i to lambda-i+2 are contracted
        theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1))  #(chi, chi, d) -> (chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi, d, chi) 
        theta = np.tensordot(theta, self.Gamma_mat[i+1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+2,:]), axes=(3,0)) #(chi, d, d, chi)
        #operator is applied, tensor is reshaped
        theta_prime = np.tensordot(theta,TimeOp[i,:,:,:,:],axes=([1,2],[2,3])) #(chi,chi,d,d)              # Two-site operator
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d*self.chi, self.d*self.chi)) #first to (d, chi, d, chi), then (d*chi, d*chi) # danger!

        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
        
        if normalize:
            self.Lambda_mat[i+1,:] = Y[:self.chi]*1/np.linalg.norm(Y[:self.chi])
        else:
            self.Lambda_mat[i+1,:] = Y[:self.chi]
        
        #truncation, and multiplication with the inverse lambda matrix of site i, where care is taken to avoid divides by 0
        X = np.reshape(X[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))  # danger!
        inv_lambdas = np.ones(self.locsize[i], dtype=complex)
        inv_lambdas *= self.Lambda_mat[i, :self.locsize[i]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i],:self.locsize[i+1]],axes=(1,1)) #(chi, d, chi)
        self.Gamma_mat[i, :, :self.locsize[i],:self.locsize[i+1]] = np.transpose(tmp_gamma,(1,0,2))
        
        #truncation, and multiplication with the inverse lambda matrix of site i+2, where care is taken to avoid divides by 0
        Z = np.reshape(Z[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = np.ones(self.locsize[i+2], dtype=complex)
        inv_lambdas *= self.Lambda_mat[i+2, :self.locsize[i+2]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(Z[:,:self.locsize[i+1],:self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0)) #(d, chi, chi)
        self.Gamma_mat[i+1, :, :self.locsize[i+1],:self.locsize[i+2]] = tmp_gamma    
        return 
     
    def TEBD(self, TimeOp, normalize):
        """ TEBD algorithm """
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp, i, normalize)
        for i in range(1, self.N-1, 2):
            self.apply_twosite(TimeOp, i, normalize)
        return
  
    def apply_MPO_locsize(self, MPO_L, MPO_M, MPO_R):
        """ Applies an MPO to the MPS """
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
        """ Calculates the expectation value of an operator Op, either for a single site or the average over the chain """
        if singlesite:
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
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
        """ Calculates the norm of the MPS """
        m_total = np.eye(chi)
        for j in range(0, self.N):        
            st = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st), st, axes=(0,0)) #(chi, chi, chi, chi)     
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
        return np.real(m_total[0,0])

    def calculate_vidal_inner(self, MPS2):
        """ Calculates the inner product of the MPS with another MPS """
        m_total = np.eye(self.chi)
        temp_lambdas, temp_gammas = MPS2.give_LG()
        for j in range(0, self.N):        
            st1 = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[j,:,:,:],np.diag(temp_lambdas[j+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st1), st2, axes=(0,0)) #(chi, chi, chi, chi)     
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
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
    return abs(temp[0,0])

def Create_Ham_MPO(J, h):
    # J*Szi*Szi+1 + h*Sz
    H_L = np.array([[-h*Sz], [Sz], [np.eye(2)]]).transpose((2,3,1,0)).reshape((2,2,1,1,3)) #transpose((1,0,2,3))
    H_M = np.array([[np.eye(2), np.zeros((2,2)), np.zeros((2,2))],[Sz, np.zeros((2,2)), np.zeros((2,2))],[-h*Sz, J*Sz, np.eye(2)]]).transpose((2,3,0,1)).reshape((2,2,1,3,3))
    H_R = np.array([np.eye(2), J*Sz, -h*Sz]).transpose((1,2,0)).reshape((2,2,1,3,1))
    return H_L, H_M, H_R

def Create_Ham(h, JXY, JZ, N, d):
    #creates XY model Hamiltonian with magnetic field h in z direction
    SX = np.kron(Sx, Sx)
    SY = np.kron(Sy, Sy)
    SZ = np.kron(Sz, Sz)
    SZ_L = np.kron(Sz, np.eye(2))
    SZ_R = np.kron(np.eye(2), Sz)
    SZ_M = (SZ_L + SZ_R)
    
    H_L = h*(SZ_L + SZ_R/2) + JXY*(SX + SY) + JZ*SZ
    H_R = h*(SZ_L/2 + SZ_R) + JXY*(SX + SY) + JZ*SZ
    H_M = h*SZ_M/2 + JXY*(SX + SY) + JZ*SZ
    
    H_arr = np.zeros((N-1, d**2, d**2), dtype=complex)
    
    for i in range(1,N-2):
        H_arr[i,:,:] = H_M
    H_arr[0,:,:] = H_L
    H_arr[N-2,:,:] = H_R
    return H_arr

def Create_TimeOp(H, delta, N, d, use_CN):
    #H = np.reshape(H, (N-1, d**2, d**2))
    U = np.ones((N-1, d**2, d**2), dtype=complex)
    
    if use_CN:
        U[0,:,:] = create_crank_nicolson(H[0], delta, N, d)
        U[N-2,:,:] = create_crank_nicolson(H[N-2], delta, N, d)
        U[1:N-2,:,:] *= create_crank_nicolson(H[1], delta, N, d) # we use broadcasting
    else:
        U[0,:,:] = expm(-1j*delta*H[0])
        U[N-2,:,:] = expm(-1j*delta*H[N-2])
        U[1:N-2,:,:] *= expm(-1j*delta*H[1]) # we use broadcasting

    U = np.around(U, decimals=15)        #Rounding out very low decimals 
    return np.reshape(U, (N-1,d,d,d,d)) 


def create_crank_nicolson(H, dt, N, d):
    H_top=np.eye(H.shape[0])-1j*dt*H/2
    H_bot=np.eye(H.shape[0])+1j*dt*H/2
    return np.linalg.inv(H_bot).dot(H_top)





####################################################################################

#### MPS constants
N=3
d=2
chi=2

#### Hamiltonian variables
h=0
JXY=-1
JZ=0
use_CN = False #choose if you want to use Crank-Nicolson approximation

#### Simulation variables
im_steps = 0
im_dt = -0.01j
steps=10
dt = 0.01
normalize = True

#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])

####################################################################################

"""
MUST LOOK INTO: continued use of locsize even as entanglement in system grows?
""" 





####################################################################################

MPS1 = MPS(N,d,chi)
#MPS1.initialize_halfstate()
MPS1.initialize_flipstate()
#MPS1.initialize_up_or_down(False)
A = MPS1.construct_superket()

temp = np.zeros((d,chi,chi))
temp[0,0,0] = np.sqrt(4/5)
temp[1,0,0] = 1/np.sqrt(5)
#MPS1.set_Gamma_singlesite(1, temp)


Ham = Create_Ham(h, JXY, JZ, N, d)


""" Ham is of shape (N,d**2,d**2), we want this to go to shape (N, d**4, d**4), so we must be careful how to use tensordot """
#conj_ham = np.reshape(np.conj(Ham), (N-1,1,d**2,d**2))
#sup_Ham = np.kron(Ham, conj_ham)[0]

#sup_Ham = np.kron(Ham, np.eye(d**2))

#sup_TimeOp = Create_TimeOp(sup_Ham, dt, N, d**2, use_CN)
#print(sup_TimeOp)


norm = np.zeros(steps)
exp_sz = np.zeros((N,steps))
TimeOp = Create_TimeOp(Ham, dt, N, d, use_CN)



def main():
    im_TimeOp = Create_TimeOp(Ham, im_dt, N, d, use_CN)
    im_norm = np.zeros(im_steps)
    im_exp_sz = np.zeros((N,im_steps))
    
    norm = np.zeros(steps)
    exp_sz = np.zeros((N,steps))
    TimeOp = Create_TimeOp(Ham, dt, N, d, use_CN)
    
    for t in range(im_steps):
        MPS1.TEBD(im_TimeOp, normalize)
            
        im_norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        #exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            im_exp_sz[j,t] = MPS1.expval(Sz, True, j)
        
    plt.plot(im_norm)
    plt.title("Normalization during im time evolution")
    plt.show()
    for j in range(N):
        plt.plot(im_exp_sz[j,:], label=f"spin {j}")
    plt.title("<Sz> per site during im time evolution")
    plt.legend()
    plt.show()
    
    for t in range(steps):
        MPS1.TEBD(TimeOp, normalize)
            
        norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        #exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            exp_sz[j,t] = MPS1.expval(Sz, True, j)
        
    plt.plot(norm)
    plt.title("Normalization during time evolution")
    plt.show()
    for j in range(N):
        plt.plot(exp_sz[j,:], label=f"spin {j}")
    plt.title("<Sz> per site during time evolution")
    plt.legend()
    plt.show()



main()































