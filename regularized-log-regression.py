import numpy as np
import pandas as pd
import math
import random
import copy
import matplotlib.pyplot as plt


def feature_mapping(X1, X2, degree=6):
    X1 = np.array(X1)
    X2 = np.array(X2)
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def plotData(X, Y):
    # Here is the grid range
    Y = np.array(np.transpose(Y))
    Y = Y.flatten()
    class_a = Y == 1
    class_b = Y == 0
    plt.plot(X[class_a, 0], X[class_a, 1], '+')
    plt.plot(X[class_b, 0], X[class_b, 1], 'o')
    plt.title('Microchips')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'], loc='upper right')


def plotDecisionBoundary(theta, X, y):

    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X, y)

    # Here is the grid range
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((u.size, v.size))
    # Evaluate z = theta*x over the grid
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            z[i, j] = np.dot(feature_mapping(ui, vj), theta)

    z = z.T  # important to transpose z before calling contour
    # print(z)

    # Plot z = 0
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    levels = [np.min(z), 0, np.max(z)]
    if not (np.min(z) < 0 < np.max(z)):
        levels.sort()
    plt.contourf(u, v, z, levels=levels, cmap='Blues', alpha=0.4)
    plt.show()


def normalize_data(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return ((array - mean) / std), mean, std


def regularized_cost_function(X, Y, theta, lambda_term):
    samples = Y.shape[0]
    h = sigmoid(np.dot(X, theta))
    cost = (1/samples) * np.sum(((np.transpose(-Y).dot(np.log(h))) -
                                 (np.transpose(1 - Y).dot(np.log(1 - h)))))
    cost += (lambda_term/2*samples) * \
        (np.transpose(theta[1:][:]).dot(theta[1:][:]))
    return cost


def gradient_descent(X, Y, theta, learning_rate, iterations, lambda_term):
    samples = Y.shape[0]
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/samples) * (X.transpose().dot(h - Y))
        theta[0, 0] = theta[0, 0] - (learning_rate) * gradient[0, 0]
        theta[1:, :] = (theta[1:, :] - (learning_rate) *
                        (gradient[1:, :] + (lambda_term/samples * theta[1:, :])))
        cost = regularized_cost_function(X, Y, theta, lambda_term)
        #print('Cost: ', cost)
    return theta, cost


def sigmoid(Z):
    Z = np.array(Z)
    result = (1 / (1 + np.exp(-Z)))
    return result


def fetch_dataset(file_name, delimiter=','):
    dataset = pd.read_csv(file_name, delimiter, header=None)
    Y = dataset.iloc[:, -1:]
    X = dataset.iloc[:, : -1]
    return np.array(X), np.array(Y)


def plot_data(X, Y, predicted=None):
    Y = np.array(np.transpose(Y))
    Y = Y.flatten()
    class_a = Y == 1
    class_b = Y == 0
    plt.plot(X[class_a, 0], X[class_a, 1], '+')
    plt.plot(X[class_b, 0], X[class_b, 1], 'o')
    plt.title('Microchips')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'], loc='upper right')
    plt.show()


def predict(features, theta, x_mean, x_std):
    features = np.subtract(features, x_mean) / x_std
    features = np.hstack([np.ones(1), features])
    return sigmoid(np.dot(features, theta))


def main():

    # fetch dataset
    X, Y = fetch_dataset('ex2data2.txt')
    n_norm = copy.deepcopy(X)

    sample_count = len(Y)

    X, x_mean, x_std = normalize_data(X)

    # feature mapping
    X = feature_mapping(
        np.array(X[:, 0]),
        np.array(X[:, 1]), degree=6)

    # norms
    feature_count = X.shape[1]
    theta = np.zeros((feature_count, 1))

    # plotting data
    # plot_data(X, Y)

    # hyperparameters
    regularizer = 1
    learning_rate = 0.1
    iterations = 10000

    # running gradient descent
    theta, cost = gradient_descent(
        X, Y, theta, learning_rate, iterations, regularizer)

    p = np.round(sigmoid(X.dot(theta)))
    print('Accuracy: %.1f %%' % (np.mean(p == Y) * 100))

    plotDecisionBoundary(theta, n_norm, Y)


if __name__ == "__main__":
    main()
