import numpy as np

import functions as f

# Setup the parameters
window_size = 8
hidden_layer_size = 10

lam = 0
learning_rate = 0.6
epochs = 500
test_size = 0.04

data = np.load("jena_processed.npy", allow_pickle=True)

# f.compare(data, lam, learning_rate, window_size, test_size, hidden_layer_size, epochs)

x_train, x_test, y_train, y_test = f.process_data(data, window_size, test_size)

initial_theta = f.initialize_model(window_size, hidden_layer_size, 1)

theta, error = f.train(x_train, x_test, y_train, y_test, initial_theta, lam, learning_rate, epochs)

f.learning_curve(epochs, error)
