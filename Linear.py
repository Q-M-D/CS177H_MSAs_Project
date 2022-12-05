import sklearn
import os
import numpy as np
import math
import random
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
# To plot pretty figures
# import pytest
import matplotlib
import matplotlib.pyplot as plt


# to make this notebook's output stable across runs

np.random.seed(42)


class Linear_Regression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.b = 0
        self.w = np.zeros(self.X.shape[1])
        self.learing_rate = 0.0001


    def update_coeffs(self):
        Y = self.Y
        Y_pred = self.predict()
        m = len(Y)
        self.b = self.b - self.learing_rate * (1 / m) * np.sum(Y_pred - Y)
        self.w = self.w - self.learing_rate * (1 / m) * np.dot((Y_pred - Y), self.X)

    def predict(self, X=np.zeros(1)):
        if X.all() == 0:
            X = self.X
        else:
            X = np.array(X)
        Y_pred = self.b + np.dot(X, self.w)
        return Y_pred

    def compute_cost(self, Y_pred):
        Y = self.Y
        m = len(Y)
        J = 1 / (2 * m) * np.sum((Y_pred - Y) ** 2)
        return J


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def get_score():
    f = open('./train_set.txt')
    data = {}
    for line in f:
        if line != '':
            tmp = line.split(' ')
            tmp[1] = float(tmp[1].replace('\n', ''))
            data[tmp[0]] = tmp[1]
    f.close()
    return data


def read_data():
    x_train = []
    y_train = []
    data = get_score()
    for item in data:
        if os.path.exists("./train_transform/" + item + ".txt"):
            file = open('./train_transform/' + item + '.txt')
            tmp = []
            for line in file:
                tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
                tmp = [float(i) for i in tmp]
            x_train.append(tmp)
            y_train.append(data[item])
    return x_train, y_train


losses = []
X, Y = read_data()
print(Y)
train_set, test_set = split_train_test(pd.DataFrame(X), 0.02)
x_train = train_set.values
y_train = Y[:len(x_train)]
x_test = test_set.values
y_test = Y[len(x_train):]


regressor = Linear_Regression(x_train, y_train)
iteration = 0
train_loss = []


while iteration < 100:
    regressor.update_coeffs()
    Y_pred = regressor.predict()
    loss = regressor.compute_cost(Y_pred)
    train_loss.append(loss)
    iteration += 1

plt.plot(train_loss)
plt.title("Loss during traning")
plt.xlabel("Iterations")
plt.ylabel("Traning loss")
plt.show()


# print Y_pred and Y_test
Y_pred = regressor.predict(x_test)
print("Y_pred: ", Y_pred)
print("Y_test", y_test)

