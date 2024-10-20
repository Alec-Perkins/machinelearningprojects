# Alec Perkins 7225998

import numpy as np
import pickle
import time

#############################################################################
print("\n1.a.ii) Mean and Covarience -------------------------------------------------")

# Instantiate the individual row vectors y1 and y2
y1 = np.array([1, 2, 3, 4, 5])
y2 = np.array([5, 4, 3, 2, 1])

# Combine the vectors into the matrix y
y = np.vstack((y1, y2))

def meanAndCov(givenMatrix):
    # Calculating the mean of matrix y: Calculating µ
    mean = np.mean(givenMatrix, axis=1)

    # Calculating the covarience of the matrix: ∑
    covMatrix = np.cov(givenMatrix, ddof=0)

    retMe = [mean, covMatrix]
    return retMe


print("Mean:\n", meanAndCov(y)[0])
print("\nCovariance matrix:\n", meanAndCov(y)[1])


#############################################################################
print("\n\n1.b) Testing Gaussian Hypothesis ----------------------------------------------")

# Opening the pickle file
with open('noisydata-1.pickle', 'rb') as pick:
    pickleData = np.array(pickle.load(pick))

print("\nNoisy Data Mean:\n", meanAndCov(pickleData)[0])
print("\nCovariance:\n", meanAndCov(pickleData)[1])


#############################################################################
print("\n\n2.a) Random nxm matrix -------------------------------------------------------")


def generateRandomMatrix(n, m):
    np.random.seed(int(time.time()))  # Seed RNG with current computer's time stamp
    randomMatrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(np.random.rand()*10)
        randomMatrix.append(row)
    return np.array(randomMatrix)

def askUserForMatrixDimension():
    n = int(input("\nWhat would you like your row size to be for your random matrix? "))
    m = int(input("\nAnd your column size? "))
    randMatrix = generateRandomMatrix(n, m)
    return randMatrix
  

usersMatrix = askUserForMatrixDimension()
print("\nHere is your random matrix:\n", usersMatrix)


#############################################################################
print("\n\n2) Singular Value Decompostion ---------------------------------------------")

def calculateMySVD(A):
    # Calculating eigenvalue decomp of A @ A^T and A^T @ A
    ATA = A.T @ A
    AAT = A @ A.T
    
    # Calculating eigen values and vectors
    eigenValuesOfU, U = np.linalg.eig(AAT)
    eigenValuesOfV, V = np.linalg.eig(ATA)
    
    # Making the zeros easier to recognize
    smallestVal = 1e-10
    eigenValuesOfU = np.where(eigenValuesOfU > smallestVal, eigenValuesOfU, 0)

    # Finding the singular values by square rooting the EVs
    singularValues = np.sqrt(eigenValuesOfU)
    Sigma = np.diag(singularValues)

    matrixAnswers = [U, Sigma, V.T]
    return matrixAnswers


# Calculating the SVD with a non-square matrix
matrixAnswers = calculateMySVD(generateRandomMatrix(5,3))
print("\nU :\n", matrixAnswers[0], "\n\n")
print("∑ :\n", matrixAnswers[1], "\n\n")
print("V :\n", matrixAnswers[2], "\n")


#############################################################################
print("\n\n3) Developing a Machine Learning Algorithm in Python with Num ----------------------")

x0 = np.array([3, 1, 3])
x1 = np.array([4, 3, 2])
x2 = np.array([5, 2, 1])
x3 = np.array([6, 1, 1])

# Creating Xp and Xf matrices or "past and future" matrices
Xp = np.vstack([x0, x1, x2]).T
Xf = np.vstack([x1, x2, x3]).T

# Printing the A matrix estimation 
A = Xf @ np.linalg.inv(Xp)
smallestVal = 1e-10
A = np.where(A > smallestVal, A, 0)

print("\nEstimated A matrix:\n", A, "\n")
