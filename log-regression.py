import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random


def fetch_dataset(file_name, delimiter=','):
    dataset = pd.read_csv(file_name, delimiter, header=None)
    Y = dataset.iloc[:, -1:]
    Y = np.array(np.transpose(Y))
    Y = Y.flatten()
    X = dataset.iloc[:, : -1]
    return np.array(X), np.array(Y)


def plot_data(X, Y, predicted=None):
    class_a = Y == 1
    class_b = Y == 0
    fig = plt.figure()
    plt.plot(X[class_a, 0], X[class_a, 1], '+')
    plt.plot(X[class_b, 0], X[class_b, 1], 'o')
    plt.title('Exam Scores')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(['Admitted', 'Not Admitted'], loc='upper right')
    plt.show()


def main():

    # fetch dataset
    X, Y = fetch_dataset('ex2data1.txt')

    sample_count = len(Y)
    feature_count = X.shape[1]

    # plotting data
    plot_data(X, Y)


if __name__ == "__main__":
    main()
