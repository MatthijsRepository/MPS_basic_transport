import os
os.chdir("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_basic_transport")

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time


class MPS:
    def __init__(self, ID, N, d, chi, is_density):
        self.ID = ID
        self.N = N
        self.d = d
        self.chi = chi
        self.is_density = is_density
        
        self.A_mat = np.zeros((N,d,chi,chi), dtype=complex) 
        #self.B_mat = np.zeros((N,d,chi,chi), dtype=complex)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        #self.canonical_site = None
        return
    
    def __str__(self):
        if self.is_density:
            return f"Density matrix {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
        else:
            return f"MPS {self.ID}, {self.N} sites of dimension {self.d}, chi={self.chi}"
    
    def give_ID(self):
        return self.ID
    
    def give_NDchi(self):
        return self.N, self.d, self.chi
    
    def give_A(self):
        """ Returns the 'A' matrices of the MPS """
        return self.A_mat
    
    def give_GL(self):
        """ Returns the Lambda and Gamma matrices of the MPS """
        return self.Gamma_mat, self.Lambda_mat
    
    def give_locsize(self):
        return self.locsize

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
    
    
    def construct_supermatrices(self, newchi):
        """ Constructs a superket of the density operator, following D. Jaschke et al. (2018) """
        sup_A_mat = np.zeros((self.N, self.d**2, newchi**2, newchi**2), dtype=complex)
        for i in range(self.N):
            sup_A_mat[i,:,:,:] = np.kron(self.A_mat[i], np.conj(self.A_mat[i]))[:newchi, :newchi]
        return sup_A_mat, np.minimum(self.locsize**2, newchi)
    
    def construct_vidal_supermatrices(self, newchi):
        """ Constructs a superket of the density operator in Vidal decomposition """
        sup_Gamma_mat = np.zeros((self.N, self.d**2, newchi, newchi), dtype=complex)
        sup_Lambda_mat = np.zeros((self.N+1, newchi))
        for i in range(self.N):
            sup_Gamma_mat[i,:,:,:] = np.kron(self.Gamma_mat[i], np.conj(self.Gamma_mat[i]))[:,:newchi,:newchi]
            sup_Lambda_mat[i,:] = np.kron(self.Lambda_mat[i], self.Lambda_mat[i])[:newchi]
        sup_Lambda_mat[N,:] = np.kron(self.Lambda_mat[N], self.Lambda_mat[N])[:newchi]
        sup_locsize = np.minimum(self.locsize**2, newchi)
        return sup_Gamma_mat, sup_Lambda_mat, sup_locsize
    
    
    def apply_singlesite(self, TimeOp, i, normalize):
        theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1))  #(chi, chi, d) -> (chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi, d, chi) 
        theta_prime = np.tensordot(theta, TimeOp, axes=(1,1)) #(chi, chi, d)
        if normalize:
            theta_prime = theta_prime / np.linalg.norm(theta_prime)
        
        inv_lambdas = np.ones(self.chi, dtype=complex)  #not working with locsize here
        inv_lambdas *= self.Lambda_mat[i]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(np.diag(inv_lambdas), theta_prime, axes=(1,0)) #(chi, chi, d) 
        
        inv_lambdas = np.ones(self.chi, dtype=complex)  #not working with locsize here
        inv_lambdas *= self.Lambda_mat[i+1]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        theta_prime = np.tensordot(theta_prime, np.diag(inv_lambdas), axes=(1,0)) #(chi, d, chi)
        self.Gamma_mat[i,:,:,:] = np.transpose(theta_prime, (1,0,2))
        return
   
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
            if self.is_density:
                self.Lambda_mat[i+1,:] = Y[:self.chi]*1/np.linalg.norm(Y[:self.chi])
                #self.Lambda_mat[i+1,:] = Y[:self.chi]
            else:   
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
     
    def TEBD(self, TimeOp, Diss_arr, normalize, Diss_bool):
        """ TEBD algorithm """
        for i in range(0, self.N-1, 2):
            self.apply_twosite(TimeOp, i, normalize)
        for i in range(1, self.N-1, 2):
            self.apply_twosite(TimeOp, i, normalize)
        
        if Diss_bool:
            for i in range(len(Diss_arr["index"])):
                self.apply_singlesite(Diss_arr["TimeOp"][i], Diss_arr["index"][i], normalize)
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
            theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d) 
            result += np.real(np.tensordot(np.conj(theta_prime),theta,axes=([0,1,2],[0,2,1])))
        return result/self.N
    
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
        temp_gammas, temp_lambdas = MPS2.give_GL()
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
    SX = np.kron(Sx, Sx)
    SY = np.kron(Sy, Sy)
    SZ = np.kron(Sz, Sz)
    SZ_L = np.kron(Sz, np.eye(2))
    SZ_R = np.kron(np.eye(2), Sz)
    SZ_M = (SZ_L + SZ_R)
    
    H_L = h*(SZ_L + SZ_R/2) + JXY*(SX + SY) + JZ*SZ
    H_R = h*(SZ_L/2 + SZ_R) + JXY*(SX + SY) + JZ*SZ
    H_M = h*SZ_M/2 + JXY*(SX + SY) + JZ*SZ
    
    H_arr = np.ones((N-1, d**2, d**2), dtype=complex)
    
    H_arr[1:N-2,:,:] *= H_M
    H_arr[0,:,:] = H_L
    H_arr[N-2,:,:] = H_R
    return H_arr
    

def Create_Dens_Ham(h, JXY, JZ, N, d):
    Sx_arr = np.array([np.kron(Sx, np.eye(d)) , np.kron(np.eye(d), Sx)])
    Sy_arr = np.array([np.kron(Sy, np.eye(d)) , np.kron(np.eye(d), Sy)])
    Sz_arr = np.array([np.kron(Sz, np.eye(d)) , np.kron(np.eye(d), Sz)])
     
    H_arr = np.ones((2, N-1, d**4, d**4), dtype=complex)
    for i in range(2):
        SX = np.kron(Sx_arr[i], Sx_arr[i])
        SY = np.kron(Sy_arr[i], Sy_arr[i])
        SZ = np.kron(Sz_arr[i], Sz_arr[i])
        SZ_L = np.kron(Sz_arr[i], np.eye(d**2))
        SZ_R = np.kron(np.eye(d**2), Sz_arr[i])
        SZ_M = (SZ_L + SZ_R)
        
        H_L = h*(SZ_L + SZ_R/2) + JXY*(SX + SY) + JZ*SZ
        H_R = h*(SZ_L/2 + SZ_R) + JXY*(SX + SY) + JZ*SZ
        H_M = h*SZ_M/2 + JXY*(SX + SY) + JZ*SZ
   
        H_arr[i, 1:N-2,:,:] *= H_M
        H_arr[i, 0,:,:] = H_L
        H_arr[i, N-2,:,:] = H_R
    
    H_eff = H_arr[0] - H_arr[1]     ######## We do not take the Hermitian conjugate into account, since H is Hermitian this has no effect
    return H_eff


def Calculate_Diss_site(L_Op, d):
    """ Creates the dissipative term for a single site """
    """ L_Op is shape (k,d,d) or (d,d) -- the k-index is in case multiple different lindblad operators act on a single site """
    if L_Op.ndim==2:
        Diss = np.kron(L_Op, np.conj(L_Op))
        Diss -= 1/2* np.kron(np.matmul(np.conj(np.transpose(L_Op)), L_Op), np.eye(d))
        Diss -= 1/2* np.kron(np.eye(d), np.matmul(np.transpose(L_Op), np.conj(L_Op)))
    else:
        Diss = np.zeros((d**2, d**2), dtype=complex)
        for i in range(np.shape(L_Op)[0]):
            Diss += np.kron(L_Op[i], np.conj(L_Op[i]))
            Diss -= 1/2* np.kron(np.matmul(np.conj(np.transpose(L_Op[i])), L_Op[i]), np.eye(d))
            Diss -= 1/2* np.kron(np.eye(d), np.matmul(np.transpose(L_Op[i]), np.conj(L_Op[i])))
    return Diss

def Create_Diss_Array(s_coup, d):
    """ Creates the array containing dissipative term, where 'index' stores the site the corresponding Lindblad operators couple to """
    Diss_arr = np.zeros((), dtype=[
        ("index", int, 2),
        ("Operator", complex, (2, d**2, d**2)),
        ("TimeOp", complex, (2, d**2, d**2))
        ])
    
    Diss_arr["index"][0] = 0
    Diss_arr["Operator"][0,:,:] = Calculate_Diss_site(np.sqrt(2*s_coup)*Sp, d)

    Diss_arr["index"][1] = N-1
    Diss_arr["Operator"][1,:,:] = Calculate_Diss_site(np.sqrt(2*s_coup)*Sm, d)
    return Diss_arr

def Calculate_Diss_TimeOp(Diss_arr, dt, d, use_CN):
    """ Calculates the dissipative time evolution operators """
    for i in range(len(Diss_arr["index"])):
        if use_CN:
            temp = create_crank_nicolson(Diss_arr["Operator"][i], dt)
        else:
            temp = expm(dt*Diss_arr["Operator"][i])
        temp = np.around(temp, decimals=15)    #Rounding out very low decimals 
        Diss_arr["TimeOp"][i,:,:] = temp
    return Diss_arr

def Create_TimeOp(H, dt, N, d, use_CN):
    #H = np.reshape(H, (N-1, d**2, d**2))
    U = np.ones((N-1, d**2, d**2), dtype=complex)
    
    if use_CN:
        U[0,:,:] = create_crank_nicolson(H[0], dt)
        U[N-2,:,:] = create_crank_nicolson(H[N-2], dt)
        U[1:N-2,:,:] *= create_crank_nicolson(H[1], dt) # we use broadcasting
    else:
        U[0,:,:] = expm(-1j*dt*H[0])
        U[N-2,:,:] = expm(-1j*dt*H[N-2])
        U[1:N-2,:,:] *= expm(-1j*dt*H[1]) # we use broadcasting

    U = np.around(U, decimals=15)        #Rounding out very low decimals 
    return np.reshape(U, (N-1,d,d,d,d)) 

def create_crank_nicolson(H, dt):
    H_top=np.eye(H.shape[0])-1j*dt*H/2
    H_bot=np.eye(H.shape[0])+1j*dt*H/2
    return np.linalg.inv(H_bot).dot(H_top)

def create_superket(State, newchi):
    """ create MPS of the density matrix of a given MPS """
    ID = State.give_ID()
    N, d, chi = State.give_NDchi()
    gammas, lambdas, locsize = State.construct_vidal_supermatrices(newchi)
    
    name = "DENS" + str(ID)
    newDENS = MPS(ID, N, d**2, newchi, True)
    newDENS.set_Gamma_Lambda(gammas, lambdas, locsize)
    globals()[name] = newDENS
    return newDENS

def create_maxmixed_normstate():
    """ Creates vectorised density matrix of an unnormalized maximally mixed state, used to calculate the trace of a vectorised density matrix """
    """ since to obtain rho11 + rho22 you must take inner [1 0 0 1] [rho11 rho12 rho21 rho22]^T without a factor 1/sqrt(2) in front """
    lambdas = np.zeros((N+1,newchi))
    lambdas[:,0]= 1
    
    gammas = np.zeros((N,d**2,newchi,newchi), dtype=complex)
    diagonal = (1+d)*np.arange(d)
    gammas[:,diagonal, 0, 0] = 1        #/2  #/np.sqrt(2)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,newchi**2)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum((d**2)**arr, newchi**2)
    
    NORM_state = MPS(0, N, d**2, newchi, True)
    NORM_state.set_Gamma_Lambda(gammas, lambdas, locsize)
    return NORM_state



####################################################################################

#### MPS constants
N=8
d=2
chi=10       #MPS truncation parameter
newchi=40   #DENS truncation parameter

#### Hamiltonian and Lindblad constants
h=0.5
JXY=1
JZ=0
s_coup = 0.3

#### Simulation variables
im_steps = 33
im_dt = -0.03j
steps=420
dt = 0.01
normalize = True
use_CN = False #choose if you want to use Crank-Nicolson approximation

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
#### Initializing simulation tools
t0 = time.time()

Ham = Create_Ham(h, JXY, JZ, N, d)
dens_Ham = Create_Dens_Ham(h, JXY, JZ, N, d)

Diss_arr = Create_Diss_Array(s_coup, d)
Diss_arr = Calculate_Diss_TimeOp(Diss_arr, dt, d, use_CN)

NORM_state = create_maxmixed_normstate()

###############
#### Simulation



MPS1 = MPS(1, N,d,chi, False)

#MPS1.initialize_halfstate()
MPS1.initialize_flipstate()
#MPS1.initialize_up_or_down(False)

#temp = np.zeros((d,chi,chi))
#temp[0,0,0] = np.sqrt(4/5)
#temp[1,0,0] = 1/np.sqrt(5)
#MPS1.set_Gamma_singlesite(1, temp)

DENS1 = create_superket(MPS1, newchi)




def main():
    im_norm = np.zeros(im_steps)
    im_exp_sz = np.zeros((N,im_steps))
    im_TimeOp = Create_TimeOp(Ham, im_dt, N, d, use_CN)
    norm = np.zeros(steps)
    exp_sz = np.zeros((N,steps))
    TimeOp = Create_TimeOp(Ham, dt, N, d, use_CN)
    
    im_dens_norm = np.zeros(im_steps)
    im_dens_exp_sz = np.zeros((N,im_steps))
    im_dens_TimeOp = Create_TimeOp(dens_Ham, im_dt, N, d**2, use_CN)
    dens_norm = np.zeros(steps)
    dens_exp_sz = np.zeros((N,steps))
    dens_TimeOp = Create_TimeOp(dens_Ham, dt, N, d**2, use_CN)
    
    for t in range(im_steps):
        if (t%10)==0:
            print(t)
        MPS1.TEBD(im_TimeOp, None, normalize, False)
        DENS1.TEBD(im_dens_TimeOp, None, normalize, False)
            
        im_norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        im_dens_norm[t] = DENS1.calculate_vidal_inner(NORM_state)
        #im_exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            im_exp_sz[j,t] = MPS1.expval(Sz, True, j)
            im_dens_exp_sz[j,t] = DENS1.expval(np.kron(Sz, np.eye(d)), True, j)
        
    plt.plot(im_norm, label="MPS")
    plt.plot(im_dens_norm, label="DENS")
    plt.title("Normalization during im time evolution")
    plt.legend()
    plt.show()
    
    for j in range(N):
        plt.plot(im_exp_sz[j,:], label=f"spin {j}")
    plt.title("MPS <Sz> per site during im time evolution")
    plt.legend()
    plt.show()
    
    for j in range(N):
        plt.plot(im_dens_exp_sz[j,:], label=f"spin {j}")
    plt.title("DENS <Sz> per site during im time evolution")
    plt.legend()
    plt.show()
    
    
    for t in range(steps):
        if (t%10)==0:
            print(t)
        MPS1.TEBD(TimeOp, None, normalize, False)
        DENS1.TEBD(dens_TimeOp, Diss_arr, normalize, False)
            
        #norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        #dens_norm[t] = DENS1.calculate_vidal_inner(NORM_state)
        #exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            exp_sz[j,t] = MPS1.expval(Sz, True, j)
            dens_exp_sz[j,t] = DENS1.expval(np.kron(Sz, np.eye(d)), True, j)
    
    plt.plot(norm, label="MPS")
    plt.plot(dens_norm, label="DENS")
    plt.title("Normalization during time evolution")
    plt.legend()
    plt.show()
    
    for j in range(N):
        plt.plot(exp_sz[j,:], label=f"spin {j}")
    plt.title("MPS <Sz> per site during time evolution")
    plt.legend()
    plt.show()
    
    for j in range(N):
        plt.plot(dens_exp_sz[j,:], label=f"spin {j}")
    plt.title("DENS <Sz> per site during time evolution")
    plt.legend()
    plt.show()
    pass



main()


test = np.zeros(N)
for i in range(N):
    test[i] = DENS1.expval(np.kron(Sz, np.eye(d)), True, i)
plt.plot(test)
plt.show()






a = MPS1.calculate_vidal_inner(MPS1)

#DENS1 = create_superket(MPS1, newchi)
c = DENS1.calculate_vidal_inner(NORM_state)

print(a)
print(c)


DENS1 = create_superket(MPS1, newchi)
d =DENS1.calculate_vidal_inner(NORM_state)
print(d)

"""
sz_test_s = MPS1.expval(Sz, True, 1)
sz_test_d1 = DENS1.expval(np.kron(Sz, np.eye(d)), True, 1)
del(DENS1)
DENS1 = create_superket(MPS1)
sz_test_d2 = DENS1.expval(np.kron(Sz, np.eye(d)), True, 1)

print()
print(sz_test_s)
print(sz_test_d1)
print(sz_test_d2)
"""


























print(time.time()-t0)