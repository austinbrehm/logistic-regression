# predict if a coffee is over $5 based on size

import numpy as np
import matplotlib.pyplot as plt
from math import e, log
import math


# data: x_train = size, y_train = true/false price is over $5
size = np.array([8, 12, 16, 20, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12])
price = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def plot_data(x, y, title, x_label, y_label):
    plt.style.use('dark_background')
    plt.scatter(x, y, marker='o', c='lime')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_sigmoid(x, w, b, title, x_label, y_label):
    plt.style.use('dark_background')
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


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        y_hat = 1 / (1 + e ** -(w * x[i] + b))
        dj_dw += (y_hat - y[i]) * x[i]
        dj_db += (y_hat - y[i])

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost_history.append(compute_cost_function(x, y, w, b))
        if i % math.ceil(iterations / 10) == 0:
            print(f'Iteration {i}: Cost {cost_history[-1]:8.2f}')
    return w, b, cost_history


# initialize w, b
# Note: change these parameters to alter model output
initial_w = 0
initial_b = 0
initial_alpha = 0.000005
initial_iterations = 10000

# compute initial cost using initial parameters
initial_cost = compute_cost_function(size, price, initial_w, initial_b)
print(f'Initial Cost Function Value: {initial_cost:0.2f}')

# compute gradient
w_gradient, b_gradient = compute_gradient(size, price, initial_w, initial_b)
print(f'dj_dw: {w_gradient:0.2f}\ndj_db: {b_gradient:0.2f}')

# compute w and b using gradient descent
w_final, b_final, cost_history_final = gradient_descent(size, price, initial_w, initial_b, initial_alpha,
                                                        initial_iterations)
print(f'w_final: {w_final:0.2f}\nb_final: {b_final:0.2f}')

final_cost = compute_cost_function(size, price, w_final, b_final)
print(f'Final Cost: {final_cost:0.2f}')

# predict if a 10 oz coffee is over $5 using w_final and b_final
size_10 = 1/(1 + e**-(w_final * 10 + b_final))
print(f'Prediction for 10 oz: {size_10:0.2f}')

if size_10 > 0.5:
    print('OVER $5.00')
else:
    print('UNDER $5.00')

# plot initial data and sigmoid function
# plot_data(size, price, 'Coffee Prices', 'size (oz)', 'price ($)')
# plot_sigmoid(size, w_final, b_final, 'Logistic Regression', 'size (Soz)', 'prediction')
