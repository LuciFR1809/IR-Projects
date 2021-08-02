import numpy
import pandas

from .config import rating_dataset


def load(filepath, column):
    """
    Reads a binary file to load dataset.
    :param
        file path 
        column Used to label the dataframe
    :return
        Pandas.Dataframe Contains the dataset.
    """
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        text = str(f.read()).strip().split('\n')
        return pandas.DataFrame.from_records(
            [sentence.split('::') for sentence in text], columns=column)


def assign_missing_values(input_matrix):
    """
    Preprocesses the input matrix to replace the NA values logically so that SVD can be performed.
    Also introduces biases to input matrix
    :param input_matrix: User-Movie rating matrix where matrix[i][j] represents rating given by
    user i for movie j
    :return preprocessed matrix which can be used for calculating SVD
    """
    matrix = numpy.asarray(input_matrix, dtype=numpy.float32)
    mean = matrix.mean()

    # Calculate biases for users and items
    row_count, col_count = [], []
    for x in range(len(input_matrix)):
        row_count.append(numpy.count_nonzero(matrix[x, :]))
    for x in range(len(matrix[0])):
        col_count.append(numpy.count_nonzero(matrix[:, x]))

    row_means, col_means = [], []
    for x in range(len(matrix)):
        row_means.append(
            (numpy.sum(matrix[x, :]) - (mean*row_count[x])) / (row_count[x] * row_count[x]))
    for x in range(len(matrix[0])):
        col_means.append(
            (numpy.sum(matrix[:, x]) - (mean*col_count[x])) / (col_count[x] * col_count[x]))

    # Replace NA values so that matrix can be used for calculating SVD
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y] == 0:
                matrix[x][y] = mean + row_means[x] + col_means[y]

            if matrix[x][y] > 5:
                matrix[x][y] = 5

            if matrix[x][y] < 1:
                matrix[x][y] = 1

    return matrix


def preprocess():
    """Wrapper function which loads dataset, preprocesses it and
    also assigns missing values to the sparse matrix

    :return
            utility_matrix numpy.ndarray containing the matrix
                                       representation of the dataset
    """
    dataset = load(rating_dataset,
                   column=['uid', 'mid', 'rating', 'time'])
    dataset.drop(labels=["time"], axis=1, inplace=True)
    dataset = dataset.astype(int)

    num_users = list(dataset['uid'].unique())
    num_users.sort()

    num_movies = list(dataset['mid'].unique())
    num_movies.sort()

    utility_matrix = numpy.full((len(num_users), len(num_movies)), 0)

    for iter in dataset.index:
        user_index = num_users.index(dataset['uid'][iter])
        movie_index = num_movies.index(dataset['mid'][iter])
        utility_matrix[user_index][movie_index] = dataset['rating'][iter]


    return assign_missing_values(utility_matrix)


if __name__ == '__main__':
    input_matrix = [[1, 1, 1, 0, 0],
                    [3, 3, 3, 0, 0],
                    [4, 4, 4, 0, 0],
                    [5, 5, 5, 0, 0],
                    [0, 2, 0, 4, 4],
                    [0, 0, 0, 5, 5],
                    [0, 1, 0, 2, 2]]

    print(assign_missing_values(input_matrix))
