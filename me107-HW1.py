import numpy as np

# Instantiate the individual row vectors y1 and y2
y1 = np.array([1, 2, 3, 4, 5])
y2 = np.array([5, 4, 3, 2, 1])

# Combine the vectors into the matrix y
y = np.vstack((y1, y2))

# Calculating the mean of matrix y: Calculating µ
mean = np.mean(y, axis=1)

# Calculating the covariance of the matrix: ∑
cov_matrix = np.cov(y, ddof=0)
print("Mean:", mean)
print("Covariance matrix:\n", cov_matrix)