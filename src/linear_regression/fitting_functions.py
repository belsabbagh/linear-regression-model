import numpy as np
import pandas as pd


def add_constant(x: np.ndarray):
    return np.append(np.ones((len(x), 1)), x, axis=1)


def least_squares(X: pd.DataFrame, y):
    """
    Return the best fit line for the given data.
    It works by finding the slope of the line of best fit and then using the slope to find the y-intercept.

    DISCLAIMER: I know it's very sensitive to outliers but it's so much faster than gradient descent.
    """
    def estimate_coefficients(x: np.ndarray, y: np.ndarray):
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    X = add_constant(X)
    return estimate_coefficients(X, y)


def mse(y_pred, y):
    return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)


def gd(X: np.ndarray, y, rate=0.05, threshold=1e-15, max_iter=None, log=False, **kwargs):

    def derivative(rate, n, x, y, y_predicted):
        """Derivative of the cost function."""
        return rate * -(2/n) * sum(x * (y-y_predicted))
    pred = kwargs.get('pred', lambda x, w: x.dot(w))
    X = add_constant(X)
    n, p = X.shape
    weights = np.zeros(p)
    prev_cost = None
    i = 1
    while prev_cost is None or diff > threshold or diff == 0:
        if max_iter is not None and i >= max_iter:
            break
        y_pred = pred(X, weights)
        weights = [w - derivative(rate, n, X.T[i], y, y_pred) for i, w in enumerate(weights)]
        cost = mse(y_pred, y)
        if log:
            print(f'{i}: {dict(cost=cost, weights=weights)}')
        diff = abs(prev_cost-cost) if prev_cost is not None else 0
        prev_cost = cost
        i += 1
    print(f"Gradient Descent: It took {i} iterations to converge.")
    return weights
