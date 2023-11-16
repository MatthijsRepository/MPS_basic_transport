import numpy as np





def initialize_halfstate(N, d, chi):
    """ Initializes the MPS into a product state of uniform eigenstates """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    Gamma_mat[:,:,0,0] = 1/np.sqrt(d)
    
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize

def initialize_flipstate(N, d, chi):
    """ Initializes the MPS into a product of alternating up/down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    for i in range(0,N,2):
        Gamma_mat[i,0,0,0] = 1
    for i in range(1,N,2):
        Gamma_mat[i,d-1,0,0] = 1
           
    #.locsize[:,:] = 1
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize

def initialize_up_or_down(N, d, chi, up):
    """ Initializes the MPS into a product state of up or down states """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    if up:  #initialize each spin in up state
        i=0 
    else:   #initialize each spin in down state
        i=d-1
    Lambda_mat[:,0] = 1
    Gamma_mat[:,i,0,0] = 1
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize


def initialize_LU_RD(N, d, chi):
    """ Initializes the MPS linearly from completely up at the leftmost site to completely down at the rightmost site """
    Lambda_mat = np.zeros((N+1,chi))
    Gamma_mat = np.zeros((N,d,chi,chi), dtype=complex)
    
    Lambda_mat[:,0] = 1
    
    temp = np.arange(N)/N
    Gamma_mat[:,0,0,0] = np.sqrt(1-temp)
    Gamma_mat[:,d-1,0,0] = np.sqrt(temp)
    
    arr = np.arange(0,N+1)
    arr = np.minimum(arr, N-arr)
    arr = np.minimum(arr,chi)               # For large L, d**arr returns negative values, this line prohibits this effect
    locsize = np.minimum(d**arr, chi)
    return Gamma_mat, Lambda_mat, locsize
    
    






