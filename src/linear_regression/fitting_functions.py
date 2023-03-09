import numpy as np
import pandas as pd


def least_squares(x: pd.DataFrame, y):
    """
    Return the best fit line for the given data.
    It works by finding the slope of the line of best fit and then using the slope to find the y-intercept.

    DISCLAIMER: I know it's very sensitive to outliers but it's so much faster than gradient descent.
    """
    def estimate_coefficients(x: np.ndarray, y: np.ndarray):
        x = np.append(np.ones((len(x), 1)), x, axis=1)
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    
    return estimate_coefficients(x, y)


def mse(y_pred, y):
    return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)


def gd(x: pd.DataFrame, y, learning_rate=0.001, threshold=1e-5, max_iter=None, log=False):

    def derivative(rate, n, x, y, y_predicted):
        """Derivative of the cost function."""
        return rate * -(2/n) * sum(x * (y-y_predicted))

    bias, weight, n = 0.01,  0.1, len(x)
    prev_cost = None
    i = 1
    while prev_cost is None or diff > threshold or diff == 0:
        if max_iter is not None and i >= max_iter:
            break
        y_pred = (weight * x) + bias
        bias -= derivative(learning_rate, n, 1, y, y_pred)
        weight -= derivative(learning_rate, n, x, y, y_pred)
        cost = mse(y_pred, y)
        if log:
            print({i: {"cost": cost, "c": bias, "m": weight}})
        diff = abs(prev_cost-cost) if prev_cost is not None else 0
        prev_cost = cost
        i += 1
    print(f"Gradient Descent: It took {i} iterations to converge.")
    return [bias, weight]
