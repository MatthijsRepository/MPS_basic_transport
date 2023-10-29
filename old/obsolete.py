
""" 

This is the twosite but modified st it follows the old method.
Useful in case you ever doubt a problem is due to the new method :)

"""
def apply_twosite_old_method(self, TimeOp, i, normalize):
        gammas = (np.ones(np.shape(self.Gamma_mat), dtype=complex) * self.Gamma_mat).transpose(0,2,3,1)
        lambdas = np.ones(np.shape(self.Lambda_mat), dtype=complex) * self.Lambda_mat
        
        theta = np.tensordot(np.diag(lambdas[i,:]), gammas[i,:,:,:], axes=(1,0)) 
        theta = np.tensordot(theta,np.diag(lambdas[i+1,:]),axes=(1,0))
        theta = np.tensordot(theta, gammas[i+1,:,:,:],axes=(2,0))
        theta = np.tensordot(theta,np.diag(lambdas[i+2,:]), axes=(2,0))     
        theta_prime = np.tensordot(theta,TimeOp[i,:,:,:,:],axes=([1,2],[2,3]))               # Two-site operator
        theta_prime = np.reshape(np.transpose(theta_prime, (2,0,3,1)),(self.d*self.chi,self.d*self.chi)) # danger!
        #Singular value decomposition
        X, Y, Z = np.linalg.svd(theta_prime); Z = Z.T
        #truncation
        if normalize:
            lambdas[i+1,:] = Y[:self.chi]*1/np.linalg.norm(Y[:self.chi])
        else:
            lambdas[i+1,:] = Y[:self.chi]
            
        X = np.reshape(X[:self.d*self.chi,:self.chi], (self.d, self.chi,self.chi))  # danger!
        inv_lambdas = np.ones(self.locsize[i], dtype=complex)
        inv_lambdas *= lambdas[i, :self.locsize[i]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        
    
        tmp_gamma = np.tensordot(np.diag(inv_lambdas),X[:,:self.locsize[i],:self.locsize[i+1]],axes=(1,1))
        gammas[i,:self.locsize[i],:self.locsize[i+1],:] = np.transpose(tmp_gamma,(0,2,1))
        
        Z = np.reshape(Z[0:self.d*self.chi,:self.chi],(self.d,self.chi,self.chi))
        Z = np.transpose(Z,(0,2,1))
        inv_lambdas = np.ones(self.locsize[i+2], dtype=complex)
        inv_lambdas *= lambdas[i+2, :self.locsize[i+2]]
        inv_lambdas[np.nonzero(inv_lambdas)] = inv_lambdas[np.nonzero(inv_lambdas)]**(-1)
        
        tmp_gamma = np.tensordot(Z[:,:self.locsize[i+1],:self.locsize[i+2]], np.diag(inv_lambdas), axes=(2,0))
        gammas[i+1,:self.locsize[i+1],:self.locsize[i+2],:] = np.transpose(tmp_gamma,(1, 2, 0))    
        #return gammas,lambdas
        gammas = gammas.transpose(0,3,1,2)
        self.Gamma_mat[:,:,:,:] = gammas[:,:,:,:]
        self.Lambda_mat[:,:] = lambdas[:,:]
        return










































    def calculate_vidal_norm(self):
        """ Calculates the norm of the MPS """
        m_total = np.eye(chi)
        for j in range(0, self.N):        
            st = np.tensordot(self.Gamma_mat[j,:,:,:],np.diag(self.Lambda_mat[j+1,:]), axes=(2,0)) #(d, chi, chi)
            mp = np.tensordot(np.conj(st), st, axes=(0,0)) #(chi, chi, chi, chi)     
            m_total = np.tensordot(m_total,mp,axes=([0,1],[0,2]))    
        return np.real(m_total[0,0])


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
        #MPS1.TEBD(im_TimeOp, None, normalize, False)
        DENS1.TEBD(im_dens_TimeOp, None, normalize, False)
            
        #im_norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        im_dens_norm[t] = DENS1.calculate_vidal_inner(NORM_state)
        #im_exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            #im_exp_sz[j,t] = MPS1.expval(Sz, True, j)
            im_dens_exp_sz[j,t] = DENS1.expval(np.kron(Sz, np.eye(d)), True, j)
        
    #plt.plot(im_norm, label="MPS")
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
        #MPS1.TEBD(TimeOp, None, normalize, False)
        DENS1.TEBD(dens_TimeOp, Diss_arr, normalize, True)
            
        #norm[t] = MPS1.calculate_vidal_inner(MPS1) #MPS1.calculate_vidal_norm()
        #dens_norm[t] = DENS1.calculate_vidal_inner(NORM_state)
        #exp_sz[t] = MPS1.expval(Sz, False, 0)
        for j in range(N):
            #exp_sz[j,t] = MPS1.expval(Sz, True, j)
            dens_exp_sz[j,t] = DENS1.expval(np.kron(Sz, np.eye(d)), True, j)
    
    #plt.plot(norm, label="MPS")
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
    
    return H_arr[0] - H_arr[1]     ######## We do not take the Hermitian conjugate into account, since H is Hermitian this has no effect



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










    def expval_old(self, Op, site):
        """ Calculates the expectation value of an operator Op for a single site """
        theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
        theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
        result = np.tensordot(theta_prime, np.conj(theta),axes=([0,1,2],[0,2,1]))
        return np.real(result)
    

    def expval_dens(self, Op, site):
        """ applies the operator and takes trace """
        theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
        theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
        theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
        
        theta_I = np.tensordot(np.diag(NORM_state.Lambda_mat[site,:]), NORM_state.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
        theta_I = np.tensordot(theta,np.diag(NORM_state.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
        return np.real(np.tensordot(theta_prime,np.conj(theta_I),axes=([0,1,2],[0,2,1])))
    
    def expval_dens_chain(self, Op):
        """ calculates expectation value for operator Op for the entire chain, using trace """
        result = np.zeros(self.N)
        for i in range(self.N):
            result[i] = self.expval_dens(Op, i)
        return result
    
    def expval_dens_old_old(self, Op, singlesite, site):
        """ applies the operator and takes trace """
        if singlesite:
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            
            theta_I = np.tensordot(np.diag(NORM_state.Lambda_mat[site,:]), NORM_state.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta_I = np.tensordot(theta,np.diag(NORM_state.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            return np.real(np.tensordot(theta_prime,np.conj(theta_I),axes=([0,1,2],[0,2,1])))
    
        else:
            result = np.zeros(self.N)      #calculate expval for entire chain
            for i in range(self.N):
                theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
                theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi,d,chi)
                theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
                
                theta_I = np.tensordot(np.diag(NORM_state.Lambda_mat[i,:]), NORM_state.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
                theta_I = np.tensordot(theta_I,np.diag(NORM_state.Lambda_mat[i+1,:]),axes=(2,0)) #(chi,d,chi)
                result[i] = np.real(np.tensordot(theta_prime, np.conj(theta_I),axes=([0,1,2],[0,2,1])))
            return result
        
        
        
    
    def expval_old_old(self, Op, singlesite, site):
        """ Calculates the expectation value of an operator Op, either for a single site or for the entire chain """
        if singlesite:
            theta = np.tensordot(np.diag(self.Lambda_mat[site,:]), self.Gamma_mat[site,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[site+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d)
            result = np.tensordot(theta_prime, np.conj(theta),axes=([0,1,2],[0,2,1]))
            return np.real(result)
 
        result = np.zeros(self.N)      #calculate expval for entire chain
        for i in range(self.N):
            theta = np.tensordot(np.diag(self.Lambda_mat[i,:]), self.Gamma_mat[i,:,:,:], axes=(1,1)) #(chi, d, chi)
            theta = np.tensordot(theta,np.diag(self.Lambda_mat[i+1,:]),axes=(2,0)) #(chi,d,chi)
            theta_prime = np.tensordot(theta, Op, axes=(1,1)) #(chi, chi, d) 
            result[i] = np.real(np.tensordot(theta_prime, np.conj(theta),axes=([0,1,2],[0,2,1])))
        return result




