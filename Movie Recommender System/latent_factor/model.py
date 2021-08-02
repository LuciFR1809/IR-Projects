import numpy
import pickle

import math

from .preprocessor import preprocess
from .config import train_bin_path, test_bin_path, validation_bin_path, utility_matrix_bin_path


class LatentFactor:
    """
    Latent Factor Model describing all the various
    factors required.
    """

    def __init__(self, alpha=0.0001, beta=0.01, k=10, epoch=30):
        """
        :param
        alpha: learning rate for stochastic gradient descent
        beta: regularisation constant for penalising magnitudes
        k: number of hidden factors used while factorising
        epoch: number of iterations performed for stochastic gradient descent
        """
        self.learning_rate = alpha
        self.regularisation_const = beta
        self.num_factors = k
        self.num_epochs = epoch

        self.num_users, self.num_items = self.load_dataset(utility_matrix_bin_path)

        self.all_ratings = self.load_dataset(train_bin_path)
        self.testing_dataset = self.load_dataset(test_bin_path)
        self.validation_dataset = self.load_dataset(validation_bin_path)

        self.global_avg_rating = numpy.mean(self.all_ratings[:, 2])

    def load_dataset(self, path):
        """Loads dataset from the binary file
        :param
            path to the binary file
        :return
            numpy.array of the dataset
        """
        with open(path, 'rb') as f:
            return numpy.array(pickle.load(f))

    def fit(self):
        """Trains our model using the training data."""
        # dimension : u X k
        user_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        item_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_items, self.num_factors))

        user_bias = numpy.zeros(self.num_users)
        item_bias = numpy.zeros(self.num_items)

        for epoch in range(self.num_epochs):
            temp_util_matrix = numpy.matmul(
                user_matrix, numpy.transpose(item_matrix))

            for user, movie, rating in self.all_ratings:

                error = (rating - temp_util_matrix[user - 1][movie - 1] -
                         self.global_avg_rating - user_bias[user - 1] - item_bias[movie - 1])

                temp_user_matrix = user_matrix[user - 1, :]

                user_matrix[user - 1, :] += self.learning_rate * (
                    error * item_matrix[movie - 1, :] - self.regularisation_const * user_matrix[user - 1, :])
                item_matrix[movie - 1, :] += self.learning_rate * (
                    error * temp_user_matrix - self.regularisation_const * item_matrix[movie - 1, :])

                user_bias[user - 1] += (self.learning_rate *
                                        (error - self.regularisation_const * user_bias[user - 1]))
                item_bias[movie - 1] += (self.learning_rate *
                                         (error - self.regularisation_const * item_bias[movie - 1]))

        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.user_bias = user_bias
        self.item_bias = item_bias

    def train(self):
        """Finds the model with least RMS, Mean-Absolute Error, 
        tested against the test dataset"""
        num_models = 1
        min_error = (10000, 10000)

        min_user_matrix = None
        min_item_matrix = None
        min_user_bias = None
        min_item_bias = None

        flag = 0
        for iter in range(num_models):
            print ("Model {}".format(iter + 1))
            self.fit()
            temp_error = (self.get_rms_error(self.testing_dataset),
                          self.get_mean_abs_error(self.testing_dataset))

            if (min(temp_error, min_error) == temp_error):
                min_user_matrix = self.user_matrix
                min_item_matrix = self.item_matrix
                min_user_bias = self.user_bias
                min_item_bias = self.item_bias

                min_error = temp_error
                flag = 1

        if (flag == 1):
            self.user_matrix = min_user_matrix
            self.item_matrix = min_item_matrix
            self.user_bias = min_user_bias
            self.item_bias = min_item_bias

    def predict(self, i, j):
        """Returns the predicted value by the model"""
        return (
            self.user_bias[i] +
            self.item_bias[j] +
            self.global_avg_rating +
            self.user_matrix[i, :].dot(self.item_matrix[j, :].T)
        )

    def get_rms_error(self, dataset):
        """Returns the Root Mean Square Error of the model"""
        error = 0
        predicted_matrix = self.get_utility_matrix()
        N = len(dataset)

        for rating_tuple in dataset:
            user, movie, rating = rating_tuple

            residual = rating - predicted_matrix[user - 1, movie - 1]
            error += pow(residual, 2)

        return math.sqrt(error/N)

    def get_mean_abs_error(self, dataset):
        """Returns the Mean Absolute Error of the model"""
        error = 0
        predicted_matrix = self.get_utility_matrix()
        N = len(dataset)

        for rating_tuple in dataset:
            user, movie, rating = rating_tuple

            residual = rating - predicted_matrix[user - 1, movie - 1]
            error += math.fabs(residual)

        return error/N

    def get_utility_matrix(self):
        """Returns the predicted utility matrix after training the model"""
        return (
            self.global_avg_rating +
            self.user_bias[:, numpy.newaxis] +
            self.item_bias[numpy.newaxis:, ] +
            self.user_matrix.dot(self.item_matrix.T))

    def __str__(self):
        """Returns the string representation of the model"""
        return str(self.get_utility_matrix())
