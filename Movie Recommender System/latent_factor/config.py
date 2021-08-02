import os

dataset_dir = "dataset"
binary_dir = "binaries"
package_dir = "latent_factor"

dataset = os.path.join(os.path.abspath('./'), dataset_dir)
movie_dataset = os.path.join(dataset, "movies.dat")
rating_dataset = os.path.join(dataset, "ratings.dat")
users_dataset = os.path.join(dataset, "users.dat")

binary = os.path.join(package_dir, binary_dir)
utility_matrix_bin_path = os.path.join(binary, "utility_matrix.pickle")
test_bin_path = os.path.join(binary, "test_data.pickle")
train_bin_path = os.path.join(binary, "train_data.pickle")
validation_bin_path = os.path.join(binary, "validation_data.pickle")
