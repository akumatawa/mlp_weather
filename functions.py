import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def flat_theta(theta):
    return np.r_[theta[0].ravel(), theta[1].ravel()]


def initialize_model(input_l, hidden_l, labels):
    epsilon = 0.12
    theta = [np.random.rand(hidden_l, input_l + 1) * 2 * epsilon - epsilon,
             np.random.rand(labels, hidden_l + 1) * 2 * epsilon - epsilon]
    return theta


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))


def forward_propagation(x, theta, m):
    ones_m = np.ones((m, 1))
    activation_vec = np.vectorize(sigmoid)

    # Layers calculation
    a1 = np.c_[ones_m, x]
    a2 = np.c_[ones_m, activation_vec(np.matmul(a1, theta[0].T))]
    a3 = np.matmul(a2, theta[1].T)
    return [a1, a2, a3]


def cost_function(x, desired, theta, lam, m):
    f_prop = forward_propagation(x, theta, m)
    arr_sub = np.subtract(f_prop[2], desired)
    arr_sum = np.power(arr_sub, 2)
    cost = np.sum(arr_sum, axis=0) / arr_sub.shape[0]

    grad = back_propagation(theta, f_prop, desired, lam, m)
    return cost, grad


def back_propagation(theta, f_prop, y_mat, lam, m):
    d1 = np.zeros(theta[0].shape)
    d2 = np.zeros(theta[1].shape)
    for i in range(0, m):
        delta_3 = np.array(f_prop[2][i] - y_mat[i])
        delta_2 = np.matmul(theta[1].T, delta_3.T)[1:].T * \
                  np.vectorize(sigmoid_grad)(np.matmul(f_prop[0][i], theta[0].T))

        d1 += np.matmul(delta_2.reshape(d1.shape[0], 1), f_prop[0][i].reshape(d1.shape[1], 1).T)
        d2 += np.matmul(delta_3.reshape(d2.shape[0], 1), f_prop[1][i].reshape(d2.shape[1], 1).T)

    theta_reg = [np.c_[np.zeros((theta[0].shape[0], 1)), theta[0][:, 1:]],
                 np.c_[np.zeros((theta[1].shape[0], 1)), theta[1][:, 1:]]]

    return np.r_[(d1 + lam * theta_reg[0]).ravel() / m,
                 (d2 + lam * theta_reg[1]).ravel() / m]


def predict(theta, x):
    m = x.shape[0]
    h1 = sigmoid(np.matmul(np.c_[np.ones((m, 1)), x], np.transpose(theta[0])).astype(float))
    h2 = np.matmul(np.c_[np.ones((m, 1)), h1], np.transpose(theta[1])).astype(float)
    return h2


def train(x_train, x_test, y_train, y_test, theta, lam, learning_rate, epochs):
    m = x_train.shape[0]
    error = []
    flatten_theta = flat_theta(theta)
    print('Training...')
    for epoch in tqdm(range(epochs)):
        cost, grad = cost_function(x_train, y_train, theta, lam, m)
        flatten_theta = flatten_theta - learning_rate * grad

        theta = [flatten_theta[0:theta[0].size].reshape(theta[0].shape),
                 flatten_theta[theta[0].size:].reshape(theta[1].shape)]

        y_train_predict = predict(theta, x_train).flatten()
        err_train_mse = np.sum(np.power((y_train_predict - y_train), 2)) / len(y_train)
        err_train_rmse = math.sqrt(err_train_mse)
        err_train_mae = np.sum(np.absolute(y_train_predict - y_train)) / len(y_train)
        err_train_mape = np.sum(np.absolute(np.divide((y_train_predict - y_train), y_train))) / len(y_train) * 100
        err_train = [err_train_mse, err_train_rmse, err_train_mae, err_train_mape]

        y_test_predict = predict(theta, x_test).flatten()
        err_test_mse = np.sum(np.power((y_test_predict - y_test), 2)) / len(y_test)
        err_test_rmse = math.sqrt(err_test_mse)
        err_test_mae = np.sum(np.absolute(y_test_predict - y_test)) / len(y_test)
        err_test_mape = np.sum(np.absolute(np.divide((y_test_predict - y_test), y_test))) / len(y_test) * 100
        err_test = [err_test_mse, err_test_rmse, err_test_mae, err_test_mape]

        error.append((err_train, err_test))

    temperature_prediction_curve(y_test_predict, y_test)

    return theta, error


def learning_curve(epochs, error):
    print(f'Train error: {error[-1][0]}')
    print(f'Test error: {error[-1][1]}')

    epoch_c = [x for x in range(epochs)][5:]
    train_acc = [x[0][1] for x in error][5:]
    test_acc = [x[1][1] for x in error][5:]

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(epoch_c, train_acc, 'r')
    ax.plot(epoch_c, test_acc, 'g')

    # set chart title and label axes.
    ax.set_title("Learning curve", fontsize=24)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)

    plt.show()


def temperature_curve(temperature):
    date = [x for x in range(len(temperature))]

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(date, temperature, 'r')

    # set chart title and label axes.
    ax.set_title("Temperature", fontsize=24)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Temperature", fontsize=14)

    plt.show()


def temperature_prediction_curve(prediction, y):
    date = [x for x in range(len(prediction))]

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(date, prediction, 'r')
    ax.plot(date, y, 'g')


    # set chart title and label axes.
    ax.set_title("Temperature", fontsize=24)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Temperature", fontsize=14)

    plt.savefig('fig.png', dpi='figure')


def process_data(in_data, window_size, test_size):
    days = len(in_data) // 24
    days = days // 5
    print(f'Processing {days} days')
    in_data = (in_data[:days*24, 1])
    in_data = (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))
    processed_data = np.empty((0, 1))
    for day in range(days):
        day_t = in_data[(day*24):((day+1)*24)]
        av_t = np.sum(day_t)/24
        processed_data = np.append(processed_data, av_t)

    # temperature_curve(in_data)
    # temperature_curve(processed_data)

    train_size = int(days * (1 - test_size))
    train_sets = train_size - window_size

    x_train = np.empty((0, window_size))
    y_train = np.empty((0, 1))
    for s in range(train_sets):
        x_train = np.append(x_train, np.reshape(processed_data[s:s + window_size], (-1, window_size)), axis=0)
        y_train = np.append(y_train, processed_data[s + window_size])

    x_test = np.empty((0, window_size))
    y_test = np.empty((0, 1))
    for s in range(train_sets, days - window_size):
        x_test = np.append(x_test, np.reshape(processed_data[s:s + window_size], (-1, window_size)), axis=0)
        y_test = np.append(y_test, processed_data[s + window_size])

    return x_train, x_test, y_train, y_test


def save_error(error, var_name, var_value):
    i = 0
    for filename in ["mse.txt", "rmse.txt", "mae.txt", "mape.txt"]:
        f = open(filename, "a")
        f.write(f"{var_name} = {var_value} : {error[i]}\n")
        f.close()
        i += 1


def append_blank_line():
    for filename in ["mse.txt", "rmse.txt", "mae.txt", "mape.txt"]:
        f = open(filename, "a")
        f.write('\n')
        f.close()


def compare(data, lam, learning_rate, window_size, test_size, hidden_layer_size, epochs):
    initial_theta = initialize_model(window_size, hidden_layer_size, 1)
    x_train, x_test, y_train, y_test = process_data(data, window_size, test_size)
    for rate in [0.5, 0.55, 0.6, 0.65, 0.7]:
        theta, error = train(x_train, x_test, y_train, y_test, initial_theta, lam, rate, epochs)
        save_error(error[-1][1], 'learnning_rate', rate)
    append_blank_line()

    for t_size in [0.01, 0.02, 0.03, 0.04, 0.05]:
        x_train, x_test, y_train, y_test = process_data(data, window_size, t_size)
        theta, error = train(x_train, x_test, y_train, y_test, initial_theta, lam, learning_rate, epochs)
        save_error(error[-1][1], 'test_size', t_size)
    append_blank_line()

    for hid_layer in [10, 14, 18, 22, 26]:
        initial_theta = initialize_model(window_size, hid_layer, 1)
        x_train, x_test, y_train, y_test = process_data(data, window_size, test_size)
        theta, error = train(x_train, x_test, y_train, y_test, initial_theta, lam, learning_rate, epochs)
        save_error(error[-1][1], "hidden_layers", hid_layer)
    append_blank_line()

    for win in [3, 4, 5, 6, 7, 8]:
        initial_theta = initialize_model(win, hidden_layer_size, 1)
        x_train, x_test, y_train, y_test = process_data(data, win, test_size)
        theta, error = train(x_train, x_test, y_train, y_test, initial_theta, lam, learning_rate, epochs)
        save_error(error[-1][1], 'window_size', win)
