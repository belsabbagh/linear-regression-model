"""
Module for models.
"""
import numpy as np


class UnfittedModelException(Exception):
    """Exception raised when prediction is attempted before fitting."""

    def __init__(self):
        super().__init__('Model is not fitted yet. Call fit() method first.')


class LinearRegression:
    """Linear regression model."""
    m: float | int | None
    c: float | int | None
    e: float | int

    def __init__(self):
        self.m = None
        self.c = None
        self.e = 0

    @staticmethod
    def mse(y_pred, y):
        return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)

    @staticmethod
    def __predict(x, m, c, e):
        if isinstance(x, (int, float)):
            return m * x + c
        return c + sum([i*m for i in x]) + e
    @staticmethod
    def best_fit(x, y):
        """Return the best fit line for the given data."""
        xbar = sum(x) / len(x)
        ybar = sum(y) / len(y)
        n = len(x)
        numer = sum([xi*yi for xi, yi in zip(x, y)]) - n * xbar * ybar
        denum = sum([xi**2 for xi in x]) - n * xbar**2
        m = numer / denum
        c = ybar - m * xbar
        return m, c

    @staticmethod
    def gradient_descent(x, y, learning_rate, threshold, log=False):

        def derivative(rate, n, x, y, y_predicted):
            """Derivative of the cost function."""
            return rate * -(2/n) * sum(x * (y-y_predicted))

        weight, bias, n = 0.1,  0.01, float(len(x))
        prev_cost = None
        i = 1
        while prev_cost is None or diff > threshold or diff == 0:
            y_pred = (weight * x) + bias
            cost = LinearRegression.mse(y_pred, y)
            weight -= derivative(learning_rate, n, x, y, y_pred)
            bias -= derivative(learning_rate, n, 1, y, y_pred)
            if log:
                print(f"{i}: Cost {cost}, Weight {weight}, Bias {bias}")
            diff = abs(prev_cost-cost) if prev_cost is not None else 0
            prev_cost = cost
            i += 1
        print(f"It took {i} iterations to converge.")
        return weight, bias

    def fit(self, x, y):
        self.m, self.c = LinearRegression.best_fit(x, y)

    def fit_gd(self, x, y, learning_rate=0.001, threshold=1e-6, log=False):
        self.m, self.c = self.gradient_descent(
            x, y, learning_rate, threshold, log=log)

    def predict(self, x):
        if self.m is None or self.c is None:
            raise UnfittedModelException()
        return self.__predict(x, self.m, self.c, self.e)
