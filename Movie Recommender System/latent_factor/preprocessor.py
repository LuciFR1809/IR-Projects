import os
import pickle
import numpy
import pandas
from sklearn.model_selection import train_test_split

from .config import movie_dataset, rating_dataset, users_dataset, \
    utility_matrix_bin_path, test_bin_path, train_bin_path, validation_bin_path


def load(filename, column):
    """
    Reads a binary file to load dataset.
    :param
        filename 
        column Used to label the dataframe
    :return
        Pandas.Dataframe Contains the dataset.
    """
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        text = str(f.read()).strip().split('\n')
        return pandas.DataFrame.from_records(
            [sentence.split('::') for sentence in text], columns=column)


def train_test_validation_split():
    """
    Splits given dataset into 70% training and 30% test dataset.
    """
    rating = load(rating_dataset, column=['uid', 'mid', 'rating', 'time'])
    rating.drop(labels=['time'], axis=1, inplace=True)

    train, test = train_test_split(rating, test_size=0.3)
    test, validation = train_test_split(test, test_size=0.5)

    return (train.astype(int), test.astype(int), validation.astype(int))

def dataset_tuple(narray, num_users, num_movies):
    t_list = []
    for index in narray.index:
        user_row = num_users.index(narray['uid'][index])
        movie_col = num_movies.index(narray['mid'][index])
        rating = int(narray['rating'][index])

        t_list.append((user_row, movie_col, rating))
    return t_list

def preprocess():
    """
    Loads dataset and generates a utility matrix (user X item ratings).
    :returns
        utility_matrix numpy.ndarray
    """
    movie = load(movie_dataset, column=['mid', 'title', 'genre'])
    movie.drop(labels=['title', 'genre'],
                axis=1, inplace=True)
    movie = movie.astype(int)

    user = load(users_dataset, column=[
                'uid', 'sex', 'age', 'occupation', 'zip-code'])
    user.drop(labels=['sex', 'age', 'occupation', 'zip-code'],
              axis=1, inplace=True)
    user = user.astype(int)


    train_data, test_data, validation_data = train_test_validation_split()

    num_users = list(user['uid'].unique())
    num_movies = list(movie['mid'].unique())
    num_users.sort()
    num_movies.sort()

    train_tuple = dataset_tuple(train_data, num_users, num_movies)
    test_tuple = dataset_tuple(test_data, num_users, num_movies)
    validation_tuple = dataset_tuple(validation_data, num_users, num_movies)

    save((len(num_users), len(num_movies)), utility_matrix_bin_path)
    save(train_tuple, train_bin_path)
    save(test_tuple, test_bin_path)
    save(validation_tuple, validation_bin_path)


def save(matrix, file_path):
    """
    Saves the a matrix in a binary file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(matrix, f)

if __name__ == "__main__":
    preprocess()
