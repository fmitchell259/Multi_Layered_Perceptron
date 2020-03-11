"""
    The following script loads the MNIST dataset
    and saves all the data, split and scaled, as
    a .npz file. This allows for much faster loading
    times while building the perceptron. 

"""

import numpy as np  


data_path ="./"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")

test_data = np.loadtxt(data_path + "mnist_test.csv",
                        delimiter=",")

# Once accessing the full files I split everything into
# training and testing labels and features. 

# At this point I scale all the values in the dataset, 
# to between -1 and 1.

X_train = ((np.asfarray(train_data[:, 1:]) / 255) - 0.5) * 2
X_test = ((np.asfarray(test_data[:, 1:]) / 255) - 0.5) * 2

# Just need the labels now and I can create a .npz file to allow
# for conveeniently and efficiently loading numpy arrays. 

y_train = np.asfarray(train_data[:, :1])
y_test = np.asfarray(test_data[:, :1])

# Now I create a .npz file to allow for easy loading of 
# the data for the rest of the project. 

np.savez_compressed('mnist_scaled.npz',
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test)



