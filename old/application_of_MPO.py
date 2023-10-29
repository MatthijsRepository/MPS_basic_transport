import numpy as np



np.random.seed(0)

W = np.arange(36).reshape(3,3,2,2)
A = np.arange(18).reshape(3,3,2)





#######################################################################################


b1 = 3
b2 = 3
a1 = 3
a2 = 3
s1 = 2
s2 = 2


"""
#######################################################################################
Dit is de fucking operatie kom maar door
"""
W = np.arange(b1*b2*s1*s2).reshape(s1, s2, 1, b1, b2)
A = np.arange(a1*a2*s2).reshape(s2, a1, a2)


test = np.kron(W,A)
testnew = np.einsum('aiibc', test)


"""
#######################################################################################
"""
print()
print(np.shape(W))
print(np.shape(A))
print(np.shape(test))
print(np.shape(testnew))




##########################################################################################

W = np.arange(b1*b2*s1*s2).reshape(b1, b2, s1, s2)
A = np.arange(a1*a2*s2).reshape(a1, a2, s2)


N1 = np.zeros((b1 * a1, b2 * a2, s1))


# Calculate the tensor product using nested loops
for i in range(b1):
    for j in range(b2):
        for k in range(a1):
            for l in range(a2):
                N1[i * a1 + k, j * a2 + l, :] = np.tensordot(W[i, j], A[k,l], axes=(1,0))

#print(N1)




W_reshaped = W.reshape(b1, b2, s1, 1, 1, s2)
A_reshaped = A.reshape(1, 1, 1, a1, a2, s2)

# Calculate the tensor product using broadcasting and tensordot
N = np.sum(W_reshaped * A_reshaped, axis=5)

# Print the result
#print(np.shape(N))

