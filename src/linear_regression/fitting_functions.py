import numpy as np


def best_fit(x, y):
    """Return the best fit line for the given data."""
    xbar, ybar = np.mean(x), np.mean(y)
    n = len(x)
    numer = sum([xi*yi for xi, yi in zip(x, y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in x]) - n * xbar**2
    m = numer / denum
    c = ybar - m * xbar
    return m, c


def mse(y_pred, y):
    return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)


def gd(x, y, learning_rate, threshold, log=False):

    def derivative(rate, n, x, y, y_predicted):
        """Derivative of the cost function."""
        return rate * -(2/n) * sum(x * (y-y_predicted))

    weight, bias, n = 0.1,  0.01, float(len(x))
    prev_cost = None
    i = 1
    while prev_cost is None or diff > threshold or diff == 0:
        y_pred = (weight * x) + bias
        cost = mse(y_pred, y)
        weight -= derivative(learning_rate, n, x, y, y_pred)
        bias -= derivative(learning_rate, n, 1, y, y_pred)
        if log:
            print(f"{i}: Cost {cost}, Weight {weight}, Bias {bias}")
        diff = abs(prev_cost-cost) if prev_cost is not None else 0
        prev_cost = cost
        i += 1
    print(f"It took {i} iterations to converge.")
    return weight, bias
