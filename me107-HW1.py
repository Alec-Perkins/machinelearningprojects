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
print("\nCovariance matrix:\n", meanAndCov(pickleData)[1])


#############################################################################
print("\n\n2.a) Random nxm matrix -------------------------------------------------------")


def generateRandomMatrix(n, m):
    np.random.seed(int(time.time()))  # Seed RNG with current computer time
    randomMatrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(round(np.random.rand()*10, 3))
        randomMatrix.append(row)
    return np.array(randomMatrix)

def askUserForMatrixDimension():
    n = int(input("\nWhat would you like your row size to be for your random matrix? "))
    m = int(input("\nAnd your column size? "))
    if type(n) is int and type(m) is int:
        randMatrix = generateRandomMatrix(n, m)
        return randMatrix
    else:
        print("Sorry, either your row input or column input was not an integer.")
        


print("\nHere is your random matrix:\n", askUserForMatrixDimension())



