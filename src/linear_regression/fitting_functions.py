import numpy as np
import pandas as pd

def slope(x,y, xbar=None, ybar=None):
    """
    Return the slope of the line of best fit.
    It works by dividing the sum of the products of the differences of the x and y values 
    by the sum of the squares of the differences of the x values.
    """
    xbar, ybar = [np.mean(x[col]) for col in x.columns] if xbar is None else xbar, np.mean(y) if ybar is None else ybar
    s = 0
    for xi, yi in zip(x.iterrows(), y):
        sx = 0
        for i, j in zip(xi[1], xbar):
            sx += i - j if isinstance(i, (int, float)) else sum([xj - j for xj in i])
        s += sx*(yi-ybar)
    # sum([(xi-xbar)*(yi-ybar) for xi, yi in zip(x, y)])/sum([(xi-xbar)**2 for xi in x])
    return s/sum([(xi[1]-xbar)**2 for xi in x.iterrows()])

def least_squares(x: pd.DataFrame, y):
    """
    Return the best fit line for the given data.
    It works by finding the slope of the line of best fit and then using the slope to find the y-intercept.
    
    DISCLAIMER: I know it's very sensitive to outliers but it's so much faster than gradient descent.
    """   
    ybar = np.mean(y)
    xbar = [np.mean(x[col]) for col in x.columns]
    m = slope(x, y, xbar, ybar)
    return [ybar - sum([s * xb for s, xb in zip(m, xbar)]), *m]


def mse(y_pred, y):
    return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)


def gd(x: pd.DataFrame, y, learning_rate=0.001, threshold=1e-6, log=False):

    def derivative(rate, n, x, y, y_predicted):
        """Derivative of the cost function."""
        return rate * -(2/n) * sum(x * (y-y_predicted))

    weight, bias, n = 0.1,  0.01, float(len(x))
    prev_cost = None
    i = 1
    while prev_cost is None or diff > threshold or diff == 0:
        y_pred = (weight * x) + bias
        weight -= derivative(learning_rate, n, x, y, y_pred)
        bias -= derivative(learning_rate, n, 1, y, y_pred)
        cost = mse(y_pred, y)
        if log:
            print({i: {"cost": cost, "m": weight, "c": bias}})
        diff = abs(prev_cost-cost) if prev_cost is not None else 0
        prev_cost = cost
        i += 1
    print(f"Gradient Descent: It took {i} iterations to converge.")
    return [bias, weight]
