import numpy as np
import matplotlib.pyplot as plt

# Practicing Calculating SVD

A = np.array([[3/4,1/4],[1/2,1/2]])

[Udiag, U] = np.linalg.eig(A@A.T)
[Vdiag, V] = np.linalg.eig(A.T@A)

print("Sigma = ", U.T@A@V)


# Math: A = USigmaV* then substitute into: U*AV = U*(UsigmaV*)V = ISigmaI

# Now doing manually:

[U_svd,S_svd,Vh_svd] = np.linalg.svd(A)
print("SVD", Vh_svd.T)