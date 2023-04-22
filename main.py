# Goal: predict if a coffee is over $5 based on size (oz).
# Logistic regression with stochastic gradient descent is used to
#   find w,b values of a sigmoid function that minimize a cost function.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import e, log


def plot(x, y, w, b, title, x_label, y_label):
    plt.style.use('dark_background')
    plt.scatter(x, y, marker='o', c='lime')
    plt.plot(x, 1 / (1 + e ** -(w * x + b)), 'fuchsia', label=f'y = 1 / (1 + e ** -({w:0.2f} * x + {b:0.2f}))')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


def plot_cost(iterations, cost_values):
    plt.style.use('dark_background')
    plt.plot(iterations, cost_values, 'fuchsia')
    plt.title('Cost vs Iteration')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()


def plot_w_b(w, b):
    plt.style.use('dark_background')
    plt.scatter(w, b, marker='o', c='lime')
    plt.title('w vs b')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.show()


def compute_cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        y_hat = 1 / (1 + e ** -(w * x[i] + b))
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
        dj_db += y_hat - y[i]

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w, b, alpha, iterations):
    cost_history = []
    w_history = []
    b_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost_history.append(compute_cost_function(x, y, w, b))
            w_history.append(w)
            b_history.append(b)
        if i % math.ceil(iterations / 10) == 0:
            print(f'Iteration {i}: Cost {cost_history[-1]:8.2f} : w {w}: b {b}')

    return w, b, cost_history, w_history, b_history


# 1. Data Collection [Note: x_train = size, y_train = binary label [true/false price is over $5]]
data = pd.read_excel('data.xlsx')
size = np.array(data['size (oz)'])
price = np.array(data['binary label'])

# 2. Initialize w, b, alpha (learning rate), and iterations [Note: change these parameters to alter model output]
initial_w = 0
initial_b = 0
initial_alpha = 0.000005
initial_iterations = 10000

# 3. Compute initial cost using initial parameters
initial_cost = compute_cost_function(size, price, initial_w, initial_b)
print(f'Initial Cost Function Value: {initial_cost:0.2f}')

# 4. Compute gradient
w_gradient, b_gradient = compute_gradient(size, price, initial_w, initial_b)
print(f'dj_dw: {w_gradient:0.2f}\ndj_db: {b_gradient:0.2f}')

# 5. Compute w and b using gradient descent
w_final, b_final, cost_history_final, w_history_final, b_history_final = gradient_descent(size, price,
                                                                                          initial_w, initial_b,
                                                                                          initial_alpha,
                                                                                          initial_iterations)
print(f'w_final: {w_final:0.2f}\nb_final: {b_final:0.2f}')

# 6. Compute final cost
final_cost = compute_cost_function(size, price, w_final, b_final)
print(f'Final Cost: {final_cost:0.2f}')

# 7. Predict if a 10 oz coffee is over $5 using w_final and b_final
size_10 = 1 / (1 + e ** -(w_final * 10 + b_final))
print(f'Prediction for 10 oz: {size_10:0.2f}')

# 8. Classification threshold
if size_10 > 0.5:
    print('OVER $5.00')
else:
    print('UNDER $5.00')

# 9. Plot initial data and sigmoid function
plot(size, price, w_final, b_final, 'Logistic Regression: Coffee Prices', 'size (oz)',
     'probability [P(coffee price > $5)] ')

plot_w_b(w_history_final, b_history_final)
plot_cost([i for i in range(1, len(cost_history_final) + 1)], cost_history_final)
