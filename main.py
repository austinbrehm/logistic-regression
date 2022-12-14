# predict if a coffee is over $5 based on size

import numpy as np
import matplotlib.pyplot as plt
from math import e, log


# data: x_train = size, y_train = true/false price is over $5
size = np.array([8, 12, 16, 20])
price = np.array([0, 0, 1, 1])


def plot(x, y, w, b, title, x_label, y_label):
    plt.style.use('dark_background')
    plt.scatter(x, y, marker='o', c='lime')
    plt.plot(x, 1/(1 + e**-(w*x + b)), 'fuchsia')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def compute_cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        y_hat = 1/(1 + e**-(w * x[i] + b))
        cost = y[i] * log(y_hat) + (1 - y[i]) * log(1 - y_hat)
    cost = cost / -m
    return cost


def compute_gradient():
    pass


def gradient_descent():
    pass


# initialize w, b
# Note: change these parameters to alter model output
initial_w = 0
initial_b = 0

# compute initial cost using initial parameters
initial_cost = compute_cost_function(size, price, initial_w, initial_b)
print(f'Initial Cost Function Value: {initial_cost:0.2f}')

# plot initial data and sigmoid function
plot(size, price, 1, 1, 'Logistic Regression: Coffee Prices Over $5.00', 'size (oz)', 'prediction')