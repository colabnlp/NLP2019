# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('home.txt', names=["size", "bedroom", "price"])

# we need to normalize the features using mean normalization
my_data = (my_data - my_data.mean()) / my_data.std()

# setting the matrixes
X = my_data.iloc[:, 0:2]
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)

y = my_data.iloc[:, 2:3].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1, 3])


# computecost
def computeCost(X, y, theta):
    tobesummed = np.power(((X @ theta.T) - y), 2)
    res = np.sum(tobesummed) / (2 * len(X))
    print("computeCost res",res)
    return res


def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        a = (X @ theta.T - y);
        b = X * a
        print(a)
        print(b)
        sum = np.sum(b, axis=0)
        print("sum",sum)

        theta = theta - (alpha / len(X)) * sum
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# set hyper parameters
alpha = 0.01
iters = 1

g, cost = gradientDescent(X, y, theta, iters, alpha)
print("g", g)

finalCost = computeCost(X, y, g)
print("finalCost", finalCost)







