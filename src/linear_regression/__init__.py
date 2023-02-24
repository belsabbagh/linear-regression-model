"""
Module for models.
"""
import numpy as np
from .fitting_functions import gd, best_fit

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
    def __predict(x, m, c, e):
        if isinstance(x, (int, float)):
            return m * x + c
        return c + sum([i*m for i in x]) + e

    def __pre_fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")

        def __type(x):
            return [x] if isinstance(x, (int, float)) else x

        return __type(x), __type(y)

    def fit(self, x, y):
        x, y = self.__pre_fit(x, y)
        self.m, self.c = best_fit(x, y)

    def fit_gd(self, x, y, learning_rate=0.001, threshold=1e-6, log=False):
        x, y = self.__pre_fit(x, y)
        self.m, self.c = gd(x, y, learning_rate, threshold, log=log)

    def predict(self, x):
        if self.m is None or self.c is None:
            raise UnfittedModelException()
        return self.__predict(x, self.m, self.c, self.e)
