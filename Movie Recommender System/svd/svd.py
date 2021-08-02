from .preprocessor import preprocess
from .model import SVD
import numpy as np


def calculate_svd(input_matrix):
    """
    Function to calculate the singular value decomposition for an input matrix
    :param input_matrix: the matrix whose svd is to be calculated
    :Return Three matrices U, S, V corresponding to the matrices after performing svd on input matrix
    """
    input_matrix = np.asarray(input_matrix, dtype=np.float32)

    U, s, Vt = SVD(input_matrix)
    return U, sigma, Vt


def calculate_svd_90(input_matrix):
    """
    Function to calculate svd for input matrix with 90% energy
    :param input_matrix the matrix whose svd is to be calculated
    :Return Three matrixes U, S, V corresponding to the matrices after performing svd with 90% energy method
    """
    input_matrix = np.asarray(input_matrix, dtype=np.float32)

    U, s, Vt = SVD(input_matrix)
    sigma = np.zeros((input_matrix.shape[0],  input_matrix.shape[1]))
    sigma[:input_matrix.shape[1], :input_matrix.shape[1]] = np.diag(s)

    total = 0
    for x in range(min(len(sigma), len(sigma[0]))):
        total = total + (sigma[x][x] * sigma[x][x])

    temp = 0
    temp_total = 0
    for x in range(min(len(sigma), len(sigma[0]))):
        temp_total = temp_total + (sigma[x][x] * sigma[x][x])
        temp = temp + 1
        if (temp_total / total) > 0.9:
            break

    new_U = U[:temp, :temp]
    new_sigma = sigma[:temp, :temp]
    new_Vt = Vt[:temp, :temp]

    print(new_U, new_sigma, new_Vt)
    return


if __name__ == "__main__":
    input_matrix = [[1, 1, 1, 0, 0],
                    [3, 3, 3, 0, 0],
                    [4, 4, 4, 0, 0],
                    [5, 5, 5, 0, 0],
                    [0, 2, 0, 4, 4],
                    [0, 0, 0, 5, 5],
                    [0, 1, 0, 2, 2]]

    preprocessed_matrix = preprocess(input_matrix)
    U, sigma, Vt = calculate_svd(preprocessed_matrix)
    print(U)
    print(sigma)
    print(Vt)
    calculate_svd_90(preprocessed_matrix)