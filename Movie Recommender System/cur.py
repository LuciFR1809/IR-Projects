from scipy.linalg import svd
import numpy as np
import math


def calculate_frob_norm(input_matrix):
    """
    Function to calculate frobenius norm values for different rows and columns
    :param input_matrix: The matrix whose frobenius norm values are to be calculated
    :Return Two tuples containing frobenius norm values for rows and columns in a pair-wise format
    """

    frob_norm_col, frob_norm_row = ([], [])
    matrix_norm = 0
    
    for i in range(len(input_matrix)):
        for j in range(len(input_matrix[0])):
            matrix_norm = matrix_norm + ((input_matrix[i][j])*(input_matrix[i][j]))
    
    for i in range(len(input_matrix)):
        row_sum = 0
        for j in range(len(input_matrix[0])):
            row_sum = row_sum + ((input_matrix[i][j])*(input_matrix[i][j]))
        frob_norm_row.append((row_sum / matrix_norm, i))

    for i in range(len(input_matrix[0])):
        col_sum = 0
        for j in range(len(input_matrix)):
            col_sum = col_sum + ((input_matrix[j][i])*(input_matrix[j][i]))
        frob_norm_col.append((col_sum / matrix_norm, i))

    frob_norm_col.sort(reverse=True)
    frob_norm_row.sort(reverse=True)

    return (frob_norm_col, frob_norm_row)


def calculate_cur(input_matrix, r):
    """
    Function to decompose input matrix using cur decomposition into three matrices
    :param input_matrix: The matrix whose decomposition is to be calculated
    :param r: r-rank approximation for input matrix
    :Return Three matrices C, U, R formed after decomposing input matrix     
    """

    frob_norm_col, frob_norm_row = calculate_frob_norm(input_matrix)

    C = []
    R = []

    for i in range(r):
        R.append(input_matrix[(frob_norm_row[i][1])])
        C.append(list(row[frob_norm_col[i][1]] for row in input_matrix))
    C = np.transpose(C)

    for i in range(len(R)):
        scale = math.sqrt(r*frob_norm_row[i][0])
        for j in range(len(R[0])):
            R[i][j] = R[i][j] / scale

    for i in range(len(C[0])):
        scale = math.sqrt(r*frob_norm_col[i][0])
        for j in range(len(C)):
            C[j][i] = C[j][i] / scale

    W = []
    for i in range(r):
        temp = []
        for j in range(r):
            temp.append(input_matrix[frob_norm_row[i][1]][frob_norm_col[j][1]])
        W.append(temp)

    # C = [[1.54, 0],
    #     [ 4.63, 0],
    #     [ 6.17, 0],
    #     [ 7.72, 0],
    #     [ 0, 9.30],
    #     [0,  11.63],
    #     [ 0, 4.65]]

    # R = [[0, 0, 0, 11.01, 11.01],
    #     [ 8.99, 8.99, 8.99, 0, 0]]

    # W = [[0, 5], 
    #     [ 5, 0]]

    W = np.asarray(W, dtype=np.float32)

    X, s, Yt = svd(W)
    sigma = np.zeros((W.shape[0],  W.shape[1]))
    sigma[:W.shape[1], :W.shape[1]] = np.diag(s)

    for i in range(len(sigma)):
        for j in range(len(sigma[0])):
            if sigma[i][j] != 0:
                sigma[i][j] = 1 / sigma[i][j]

    Y = np.transpose(Yt)
    Xt = np.transpose(X)
    U = np.dot(Y, np.dot(np.dot(sigma, sigma), Xt))

    return (C, U, R)


if __name__ == "__main__":
    input_matrix = [[1, 1, 1, 0, 0],
                    [3, 3, 3, 0, 0],
                    [4, 4, 4, 0, 0],
                    [5, 5, 5, 0, 0],
                    [0, 0, 0, 4, 4],
                    [0, 0, 0, 5, 5],
                    [0, 0, 0, 2, 2]]

    C, U, R = calculate_cur(input_matrix, 4)
    result = np.matmul(C, np.matmul(U, R))

    print(result)

    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] > 5:
                result[i][j] = 0
            elif result[i][j] < 0:
                if ((abs(result[i][j]) > 0) and (abs(result[i][j]) < 5)):
                    result[i][j] = abs(result[i][j])
                else:
                    result[i][j] = 0 

    print(result)