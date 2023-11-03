import os
os.chdir("C:\\Users\\matth\\OneDrive\\Documents\\TUDelft\\MEP\\code\\MPS_basic_transport")

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import pickle
import time
from datetime import datetime


class MPS:
    def __init__(self, ID, N, d, chi, is_density):
        self.ID = ID
        self.N = N
        self.d = d
        self.chi = chi
        self.is_density = is_density
        if is_density:
            self.name = "DENS"+str(ID)+"_N"+str(N)+"_chi"+str(chi)
        else: 
            self.name = "MPS"+str(ID)+"_N"+str(N)+"_chi"+str(chi)
        
        self.A_mat = np.zeros((N,d,chi,chi), dtype=complex) 
        #self.B_mat = np.zeros((N,d,chi,chi), dtype=complex)
        
        self.Lambda_mat = np.zeros((N+1,chi))
        self.Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)

        self.locsize = np.zeros(N+1, dtype=int)     #locsize tells us which slice of the matrices at each site holds relevant information
        #self.canonical_site = None
        return
    
    def store(self):
        time = str(datetime.now())
        timestr = time[5:7] + time[8:10] + "_" + time[11:13] + time[14:16] + "_"  #get month, day, hour, minute
        
        folder = "data\\" 
        filename = timestr + self.name + ".pkl"
        
        file = open(folder + filename, 'wb')
        pickle.dump(self, file)
        
        print(f"Stored {filename} to memory")
        pass
    
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
        """ Applies a single-site operator to site i """
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
        
        ##USE lambdas[x].copy()
        
        inv_lambdas = np.ones(self.locsize[i], dtype=complex)
        inv_lambdas *= self.Lambda_mat[i, :self.locsize[i]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i],:self.locsize[i+1]],axes=(1,1)) #(chi, d, chi)
        self.Gamma_mat[i, :, :self.locsize[i],:self.locsize[i+1]] = np.transpose(tmp_gamma,(1,0,2))
        
        #truncation, and multiplication with the inverse lambda matrix of site i+2, where care is taken to avoid divides by 0
        Z = np.reshape(Z[:self.d*self.chi, :self.chi], (self.d, self.chi, self.chi))
        Z = np.transpose(Z,(0,2,1))
        
        ##USE lambdas[x].copy()
        
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
    
    def expval(self, Op, site):
        """ Calculates the expectation value of an operator Op for a single site """
        theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
        theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
        """
        #NOTE:  for singlesite expectations the expectation <<p |A| p>> matches the result for pure state evolutions
        #        hence that specific expectation should be used
        if 1==2:#self.is_density:     #In case of density matrices we must take the trace           
            theta_I = NORM_state.singlesite_thetas
            #theta_I = np.tensordot(np.diag(NORM_state.Lambda_mat[site,:]), NORM_state.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            #theta_I = np.tensordot(theta_I,np.diag(NORM_state.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            return np.real(np.tensordot(theta_prime, np.conj(theta_I), axes=([0,1,2],[0,2,1])))
        else:
        """
        return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2],[0,2,1])))
    
    def expval_chain(self, Op):
        """ calculates expectation value for operator Op for the entire chain """
        result = np.zeros(self.N)
        for i in range(self.N):
            result[i] = self.expval(Op, i)
        return result
        
    def expval_twosite(self, Op, site):
        """ Calculates expectation value for a twosite operator Op at sites site and site+1 """
        theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1))  #(chi, chi, d) -> (chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi, d, chi) 
        theta = np.tensordot(theta, self.Gamma_mat[site+1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+2,:]), axes=(3,0)) #(chi, d, d, chi)
        Op = np.reshape(Op, (self.d,self.d,self.d,self.d))
        theta_prime = np.tensordot(theta, Op,axes=([1,2],[2,3])) #(chi,chi,d,d) 
        """
        if 1==2:#self.is_density:
            theta_I = NORM_state.twosite_thetas
            #theta_I = np.tensordot(np.diag(NORM_state.Lambda_mat[site,:]), NORM_state.Gamma_mat[site,:,:,:], axes=(1,1))  #(chi, chi, d) -> (chi, d, chi)
            #theta_I = np.tensordot(theta_I,np.diag(NORM_state.Lambda_mat[site+1,:]),axes=(2,0)) #(chi, d, chi) 
            #theta_I = np.tensordot(theta_I, NORM_state.Gamma_mat[site+1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
            #theta_I = np.tensordot(theta_I,np.diag(NORM_state.Lambda_mat[site+2,:]), axes=(3,0)) #(chi, d, d, chi)
            return np.tensordot(theta_prime, theta_I, axes=([0,1,2,3],[0,3,1,2]))
        else:
            #pass
        """
        return np.real(np.tensordot(theta_prime, np.conj(theta), axes=([0,1,2,3],[0,3,1,2])))

        
    def calculate_energy(self, TimeEvol_obj):
        """ calculate the energy of the entire chain from a given Hamiltonian """
        Energy = 0
        for i in range(self.N-1):
            Energy += self.expval_twosite(TimeEvol_obj.Ham_energy[i],i)
        return Energy
    
    def calculate_vidal_inner(self, MPS2):
        """ Calculates the inner product of the MPS with another MPS """
        m_total = np.eye(self.chi)
        temp_gammas, temp_lambdas = MPS2.give_GL()  #retrieve gammas and lambdas of MPS2
        for j in range(0, self.N):        
            st1 = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            st2 = np.tensordot(temp_gammas[j,:,:,:],np.diag(temp_lambdas[j+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st1), st2, axes=(0,0)) #(chi, chi, chi, chi)     
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
        return abs(m_total[0,0])
    
    def time_evolution(self, TimeEvol_obj, normalize, steps, desired_expectations, track_normalization, track_energy):
        if TimeEvol_obj.is_density != self.is_density:
            print("Error: time evolution operator type does not match state type (MPS/DENS)")
            return
        
        if track_normalization:
            Normalization = np.zeros(steps)
        if track_energy:
            energy = np.zeros(steps)
        exp_values = np.ones((len(desired_expectations), self.N, steps)) #array to store expectation values in
        
        if steps>current_cutoff:
            spin_current_values = np.zeros(steps-current_cutoff)
        
        TimeOp = TimeEvol_obj.TimeOp
        Diss_arr = TimeEvol_obj.Diss_arr
        Diss_bool = TimeEvol_obj.Diss_bool

        
        print(f"Starting time evolution of {self.name}")
        for t in range(steps):
            if (t%20==0):
                print(t)
            
            if track_normalization:
                if self.is_density:
                    Normalization[t] = self.calculate_vidal_inner(NORM_state)
                else: 
                    Normalization[t] = self.calculate_vidal_inner(self)
            if track_energy:
                energy[t] = self.calculate_energy(TimeEvol_obj)
            
            for i in range(len(desired_expectations)):
                if desired_expectations[i][2] == True:
                    exp_values[i,:,t] *= self.expval(desired_expectations[i][1], desired_expectations[i][3])
                else:
                    exp_values[i,:,t] *= self.expval_chain(desired_expectations[i][1])
                                    
            self.TEBD(TimeOp, Diss_arr, normalize, Diss_bool)
                        
            if (t>current_cutoff and Diss_bool==True):
                spin_current_values[t-current_cutoff] = self.expval_twosite(spin_current_op, round(self.N/2))
        
        #### Plotting takes place here
        time_axis = np.arange(steps)*abs(TimeEvol_obj.dt)
        
        if track_normalization:
            plt.plot(Normalization)
            plt.title(f"Normalization of {self.name} over time")
            plt.xlabel("Time")
            plt.xlabel("Normalization")
            plt.grid()
            plt.show()
        if track_energy:
            plt.plot(time_axis, energy)
            plt.title(f"Energy of {self.name}")
            plt.xlabel("Time")
            plt.ylabel("Energy")
            plt.grid()
            plt.show()

        for i in range(len(desired_expectations)):
            if desired_expectations[i][2]==False:
                for j in range(self.N):
                    plt.plot(time_axis, exp_values[i,j,:], label=f"Site {j}")
            else:
                plt.plot(time_axis, exp_values[i,0,:], label=f"Site {desired_expectations[i][3]}")
            plt.xlabel("Time")
            plt.ylabel(f"<{desired_expectations[i][0]}>")
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            #plt.legend()
            plt.title(f"Plot of <{desired_expectations[i][0]}> of {self.name} over time")
            plt.grid()
            plt.show()
                   
        print("Time averaged spin current through middle site:")
        print((np.average(spin_current_values)))
        print("Imaginary part of spin current:")
        print(np.imag(np.average(spin_current_values)))
        return

########################################################################################  

class Time_Operator:
    def __init__(self,N, d, JXY, JZ, h, s_coup, dt, Diss_bool, is_density, use_CN):
        self.N = N
        self.d = d
        self.JXY = JXY
        self.JZ = JZ
        self.h = h
        self.s_coup = s_coup
        self.dt = dt
        self.is_density = is_density
        self.Diss_bool = Diss_bool
        self.use_CN = use_CN
        
        if isinstance(dt, complex):     
            self.is_imaginary = True
        else:
            self.is_imaginary = False
        
        if self.is_density==False:
            self.Diss_bool=False
       
        #### Creating Hamiltonian and Time operators
        #### Note: Ham_energy is the Hamiltonian to be used for energy calculation
        if self.is_density:
            self.Ham, self.Ham_energy = self.Create_Dens_Ham()
        else:
            self.Ham = self.Create_Ham()
            self.Ham_energy = self.Ham
        
        self.TimeOp = self.Create_TimeOp(self.dt, self.use_CN)
        
        if (self.is_density and self.Diss_bool):
            self.Diss_arr = self.Create_Diss_Array(self.s_coup)
            self.Calculate_Diss_TimeOp(self.dt, self.use_CN)
        else:
            self.Diss_arr = None
        return
        
    def Create_Ham(self):
        """ Create Hamiltonian for purestate """
        SX = np.kron(Sx, Sx)
        SY = np.kron(Sy, Sy)
        SZ = np.kron(Sz, Sz)
        SZ_L = np.kron(Sz, np.eye(self.d))
        SZ_R = np.kron(np.eye(self.d), Sz)
        SZ_M = (SZ_L + SZ_R)
        
        H_L = self.h*(SZ_L + SZ_R/2) + self.JXY*(SX + SY) + self.JZ*SZ
        H_R = self.h*(SZ_L/2 + SZ_R) + self.JXY*(SX + SY) + self.JZ*SZ
        H_M = self.h*SZ_M/2 + self.JXY*(SX + SY) + self.JZ*SZ
        
        H_arr = np.ones((self.N-1, self.d**2, self.d**2), dtype=complex)
        
        H_arr[1:self.N-2,:,:] *= H_M
        H_arr[0,:,:] = H_L
        H_arr[self.N-2,:,:] = H_R
        return H_arr
        
    def Create_Dens_Ham(self):
        """ create effective Hamiltonian for time evolution of the density matrix """
        Sx_arr = np.array([np.kron(Sx, np.eye(self.d)) , np.kron(np.eye(self.d), Sx)])
        Sy_arr = np.array([np.kron(Sy, np.eye(self.d)) , np.kron(np.eye(self.d), Sy)])
        Sz_arr = np.array([np.kron(Sz, np.eye(self.d)) , np.kron(np.eye(self.d), Sz)])
         
        H_arr = np.ones((2, N-1, self.d**4, self.d**4), dtype=complex)
        for i in range(2):
            SX = np.kron(Sx_arr[i], Sx_arr[i])
            SY = np.kron(Sy_arr[i], Sy_arr[i])
            SZ = np.kron(Sz_arr[i], Sz_arr[i])
            SZ_L = np.kron(Sz_arr[i], np.eye(self.d**2))
            SZ_R = np.kron(np.eye(self.d**2), Sz_arr[i])
            SZ_M = (SZ_L + SZ_R)
            
            H_L = self.h*(SZ_L + SZ_R/2) + self.JXY*(SX + SY) + self.JZ*SZ
            H_R = self.h*(SZ_L/2 + SZ_R) + self.JXY*(SX + SY) + self.JZ*SZ
            H_M = self.h*SZ_M/2 + self.JXY*(SX + SY) + self.JZ*SZ
       
            H_arr[i, 1:self.N-2,:,:] *= H_M
            H_arr[i, 0,:,:] = H_L
            H_arr[i, self.N-2,:,:] = H_R

        #Note: H_arr[0] is the correct Hamiltonian to use for energy calculations
        return (H_arr[0] - np.conj(H_arr[1])), H_arr[0]     

    def Create_TimeOp(self, dt, use_CN):
        if self.is_density:
            U = np.ones((self.N-1, self.d**4, self.d**4), dtype=complex)
        else:
            U = np.ones((self.N-1, self.d**2, self.d**2), dtype=complex)
        
        if use_CN:
            U[0,:,:] = self.create_crank_nicolson(self.Ham[0], dt)
            U[self.N-2,:,:] = self.create_crank_nicolson(self.Ham[self.N-2], dt)
            U[1:self.N-2,:,:] *= self.create_crank_nicolson(self.Ham[1], dt) # we use broadcasting
        else:
            U[0,:,:] = expm(-1j*dt*self.Ham[0])
            U[self.N-2,:,:] = expm(-1j*dt*self.Ham[self.N-2])
            U[1:self.N-2,:,:] *= expm(-1j*dt*self.Ham[1]) # we use broadcasting
    
        U = np.around(U, decimals=15)        #Rounding out very low decimals 
        if self.is_density:
            return np.reshape(U, (self.N-1,self.d**2,self.d**2,self.d**2,self.d**2))
        else:
            return np.reshape(U, (self.N-1,self.d,self.d,self.d,self.d)) 

    def create_crank_nicolson(self, H, dt):
        """ Creates the Crank-Nicolson operator from a given Hamiltonian """
        H_top=np.eye(H.shape[0])-1j*dt*H/2
        H_bot=np.eye(H.shape[0])+1j*dt*H/2
        return np.linalg.inv(H_bot).dot(H_top)

    def Calculate_Diss_site(self, Lind_Op):
        """ Creates the dissipative term for a single site """
        """ Lind_Op is shape (k,d,d) or (d,d) -- the k-index is in case multiple different lindblad operators act on a single site """
        Diss = np.zeros((self.d**2, self.d**2), dtype=complex)
        if Lind_Op.ndim==2:     #If only a single operator is given, this matrix is used
            Diss += np.kron(Lind_Op, np.conj(Lind_Op))
            Diss -= 1/2* np.kron(np.matmul(np.conj(np.transpose(Lind_Op)), Lind_Op), np.eye(self.d))
            Diss -= 1/2* np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op), np.conj(Lind_Op)))
        else:                   #If multiple matrices are given, the sum of Lindblad operators is used
            for i in range(np.shape(Lind_Op)[0]):
                Diss += np.kron(Lind_Op[i], np.conj(Lind_Op[i]))
                Diss -= 1/2* np.kron(np.matmul(np.conj(np.transpose(Lind_Op[i])), Lind_Op[i]), np.eye(self.d))
                Diss -= 1/2* np.kron(np.eye(self.d), np.matmul(np.transpose(Lind_Op[i]), np.conj(Lind_Op[i])))
        return Diss
    
    def Create_Diss_Array(self, s_coup):
        """ Creates the array containing dissipative term, where 'index' stores the site the corresponding Lindblad operators couple to """
        Diss_arr = np.zeros((), dtype=[
            ("index", int, 2),
            ("Operator", complex, (2, self.d**2, self.d**2)),
            ("TimeOp", complex, (2, self.d**2, self.d**2))
            ])
        
        Diss_arr["index"][0] = 0
        Diss_arr["Operator"][0,:,:] = self.Calculate_Diss_site(s_coup*Sp)
    
        #Diss_arr["index"][1] = N-1
        #Diss_arr["Operator"][1,:,:] = self.Calculate_Diss_site(np.sqrt(2*s_coup)*np.eye(self.d))
    
        Diss_arr["index"][1] = N-1
        Diss_arr["Operator"][1,:,:] = self.Calculate_Diss_site(s_coup*Sm)
        return Diss_arr
    
    def Calculate_Diss_TimeOp(self, dt, use_CN):
        """ Calculates the dissipative time evolution operators """
        for i in range(len(self.Diss_arr["index"])):
            if use_CN:
                temp = self.create_crank_nicolson(self.Diss_arr["Operator"][i], dt)
            else:
                temp = expm(dt*self.Diss_arr["Operator"][i])
            temp = np.around(temp, decimals=15)    #Rounding out very low decimals 
            self.Diss_arr["TimeOp"][i,:,:] = temp
        return



###################################################################################

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


def calculate_thetas_singlesite(state):
    """ contracts lambda_i gamma_i lambda_i+1 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    #Note, the lambda matrices are just a factor 1, it is possible to simply return a reshaped gamma matrix
    #temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    #return np.tensordot(temp, np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi,d,chi)
    return state.Gamma_mat[0].transpose(0,2,1)

def calculate_thetas_twosite(state):
    """ contracts lambda_i gamma_i lambda_i+1 gamma_i+1 lambda_i+2 (:= theta) for each site and returns them, used for the NORM_state """
    """ NOTE: only works for NORM_state since there the result is the same for all sites! """
    """ This function is used to prevent redundant calculation of these matrices """
    temp = np.tensordot(np.diag(state.Lambda_mat[0,:]), state.Gamma_mat[0,:,:,:], axes=(1,1)) #(chi, d, chi)
    temp = np.tensordot(temp,np.diag(state.Lambda_mat[1,:]),axes=(2,0)) #(chi, d, chi) 
    temp = np.tensordot(temp, state.Gamma_mat[1,:,:,:],axes=(2,1)) #(chi, d, chi, d) -> (chi,d,d,chi)
    return np.tensordot(temp,np.diag(state.Lambda_mat[2,:]), axes=(3,0)) #(chi, d, d, chi)



####################################################################################
t0 = time.time()
#### MPS constants
N=10
d=2
chi=10       #MPS truncation parameter
newchi=25   #DENS truncation parameter


#### Hamiltonian and Lindblad constants
h=0
JXY=1#1
JZ=1

s_coup=1
s_coup = np.sqrt(2*s_coup)  

#### Simulation variables
im_steps = 0
im_dt = -0.03j

steps=1000
dt = 0.02
cutoff_factor = 0
current_cutoff=round(steps * cutoff_factor) #<-----

normalize = False
use_CN = False #choose if you want to use Crank-Nicolson approximation
Diss_bool = True

#### Spin matrices
Sp = np.array([[0,1],[0,0]])
Sm = np.array([[0,0],[1,0]])
Sx = np.array([[0,1], [1,0]])
Sy = np.array([[0,-1j], [1j,0]])
Sz = np.array([[1,0],[0,-1]])


NORM_state = create_maxmixed_normstate()
NORM_state.singlesite_thetas = calculate_thetas_singlesite(NORM_state)
NORM_state.twosite_thetas = calculate_thetas_twosite(NORM_state)



#Operator for calculating spin current from Prosen 2009.
spin_current_op = np.kron( np.kron(Sx, np.eye(d)) , np.kron(Sy, np.eye(d))) - np.kron( np.kron(Sy, np.eye(d)) , np.kron(Sx, np.eye(d)))

#equivalent operator in terms of Sp and Sm
#spin_current_op = 2*1j* ( np.kron( np.kron(Sp, np.eye(d)) , np.kron(Sm, np.eye(d))) - np.kron( np.kron(Sm, np.eye(d)) , np.kron(Sp, np.eye(d))) )

####################################################################################

    
#temp = np.zeros((d,chi,chi))
#temp[0,0,0] = np.sqrt(4/5)
#temp[1,0,0] = 1/np.sqrt(5)
#MPS1.set_Gamma_singlesite(0, temp)

def main():
    MPS1 = MPS(1, N,d,chi, False)
    MPS1.initialize_halfstate()
    #MPS1.initialize_flipstate()
    #MPS1.initialize_up_or_down(False)    
    
    
    DENS1 = create_superket(MPS1, newchi)
    TimeEvol_obj1 = Time_Operator(N, d, JXY, JZ, h, s_coup, dt, Diss_bool, True, use_CN)
    
    desired_expectations = []
    #desired_expectations.append(("I", np.eye(d**2), False, 0))
    desired_expectations.append(("Sz", np.kron(Sz, np.eye(d)), False, 0))
    #desired_expectations.append(("Sz", np.kron(Sz, np.eye(d)), True, 1))
    
    DENS1.time_evolution(TimeEvol_obj1, normalize, steps, desired_expectations, True, False)


    #TimeEvol_obj2 = Time_Operator(N, d, JXY, JZ, h, s_coup, dt, False, False, use_CN)
    #pure_desired_expectations = []
    #pure_desired_expectations.append(("Sz", Sz, False, 0))
    #MPS1.time_evolution(TimeEvol_obj2, normalize, steps, pure_desired_expectations, False, False)
    #DENS2 = create_superket(MPS1, newchi)
    
   
    """
    final_Sz = np.zeros(N)
    for i in range(N):
        final_Sz[i] = DENS1.expval(np.kron(Sz, np.eye(d)), True, i)
    plt.plot(final_Sz, linestyle="", marker=".")
    plt.xlabel("Site")
    plt.ylabel("<Sz>")
    plt.grid()
    plt.title(f"<Sz> for each site after {steps} steps with dt={dt}")
    plt.show()    
    """
    
    pass

main()





#####
# N_list = [5, 10]
# J_list = [0.3342275264055715, 0.2090220151776465]




"""
a = MPS1.calculate_vidal_inner(MPS1)

#DENS1 = create_superket(MPS1, newchi)
c = DENS1.calculate_vidal_inner(NORM_state)

print(a)
print(c)


DENS1 = create_superket(MPS1, newchi)
d =DENS1.calculate_vidal_inner(NORM_state)
print(d)

""" 





















elapsed_time = time.time()-t0
print(f"Elapsed simulation time: {elapsed_time}")