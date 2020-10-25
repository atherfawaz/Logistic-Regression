import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_data(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return ((array - mean) / std), mean, std


def cost_function(X, Y, theta):
    samples = Y.shape[0]
    h = sigmoid(np.dot(X, theta))
    cost = (1 / samples) * np.sum(((np.transpose(-Y).dot(np.log(h))) -
                                   (np.transpose(1 - Y).dot(np.log(1 - h)))))
    return cost


def gradient_descent(X, Y, theta, learning_rate, iterations):
    samples = Y.shape[0]
    cost = 0
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        theta = theta - (learning_rate / samples) * (X.transpose().dot(h - Y))
        cost = cost_function(X, Y, theta)
    return theta, cost


def sigmoid(Z):
    Z = np.array(Z)
    return (1 / (1 + np.exp(-Z)))


def fetch_dataset(file_name, delimiter=','):
    dataset = pd.read_csv(file_name, delimiter, header=None)
    Y = dataset.iloc[:, -1:]
    X = dataset.iloc[:, : -1]
    return np.array(X), np.array(Y)


def plot_data(X, Y, predicted=False):
    Y = np.array(np.transpose(Y))
    Y = Y.flatten()
    class_a = Y == 1
    class_b = Y == 0
    if predicted:
        plt.plot(X[class_a, 1], X[class_a, 2], '+')
        plt.plot(X[class_b, 1], X[class_b, 2], 'o')
    else:
        plt.plot(X[class_a, 0], X[class_a, 1], '+')
        plt.plot(X[class_b, 0], X[class_b, 1], 'o')
    plt.title('Exam Scores')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(['Admitted', 'Not Admitted'], loc='upper right')
    if not predicted:
        plt.show()


def predict(features, theta, x_mean, x_std):
    features = np.subtract(features, x_mean) / x_std
    features = np.hstack([np.ones(1), features])
    return sigmoid(np.dot(features, theta))


def plot_decision_boundary(X, Y, theta):
    plot_data(X, Y, predicted=True)
    plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
    plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y)
    plt.legend(['Admitted', 'Not admitted',
                'Decision Boundary'], loc='upper right')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()


def main():
    # fetch dataset
    X, Y = fetch_dataset('ex2data1.txt')
    X_original = copy.deepcopy(X)

    sample_count = len(Y)
    feature_count = X.shape[1]

    # plotting data
    plot_data(X, Y)

    # thetas
    X, x_mean, x_std = normalize_data(X)
    X = np.hstack([np.ones((sample_count, 1)), X])
    theta = np.random.rand(feature_count + 1, 1)

    # hyperparameters
    learning_rate = 0.5
    iterations = 2000

    # running gradient descent
    theta, cost = gradient_descent(X, Y, theta, learning_rate, iterations)

    print('Theta: ', theta)
    print('Cost: ', cost)

    # calculating for a given value
    features = np.array([45, 85])
    prediction = predict(features, theta, x_mean, x_std)
    print('For a student with a score of 45 in exam 1 and 85 in exam 2, the probability of admission is: ', prediction)

    # plot decision boundary
    plot_decision_boundary(X, Y, theta)


if __name__ == "__main__":
    main()
