"""
Module for models.
"""
import numpy as np
from .fitting_functions import gd, least_squares

class UnfittedModelException(Exception):
    """Exception raised when prediction is attempted before fitting."""

    def __init__(self):
        super().__init__('Model is not fitted yet. Call fit() method first.')


class LinearRegression:
    """Linear regression model."""
    m: float | int | None = None
    c: float | int | None = None

    def __init__(self):
        return None

    @staticmethod
    def __predict(x, m, c):
        if isinstance(x, (int, float)):
            return m * x + c
        return c + sum([i*m for i in x])

    def _pre_fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")

        def __type(x):
            return [x] if isinstance(x, (int, float)) else x

        return __type(x), __type(y)

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        self.m, self.c = least_squares(x, y)

    def predict(self, x):
        if self.m is None or self.c is None:
            raise UnfittedModelException()
        return self.__predict(x, self.m, self.c)


class GDLinearRegression(LinearRegression):
    learning_rate: float | int = 0.001
    threshold: float | int = 1e-6
    def __init__(self, learning_rate=0.001, threshold=1e-6):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0.")
        self.learning_rate = learning_rate
        self.threshold = threshold
        
    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        coeffs = gd(x, y, self.learning_rate, self.threshold)
        self.c, self.m = coeffs[0], coeffs[1]