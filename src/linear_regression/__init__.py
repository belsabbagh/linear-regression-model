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
    W: list[float | int] | None = None
    b: float | int | None = None

    def __init__(self):
        return None

    @staticmethod
    def __predict(x, m, c):
        if isinstance(x, (int, float)):
            return m * x + c
        return c + np.dot(x, m)

    def _pre_fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")

        def __type(x):
            return [x] if isinstance(x, (int, float)) else x

        return __type(x), __type(y)

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        res = least_squares(x, y)
        self.b, self.W = res[0], res[1:]

    def predict(self, x):
        if self.W is None or self.b is None:
            raise UnfittedModelException()
        return self.__predict(x, self.W, self.b)


class GDLinearRegression(LinearRegression):
    rate: float | int = 0.05
    threshold: float | int = 1e-15

    def __init__(self, learning_rate=0.05, threshold=1e-15):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0.")
        self.rate = learning_rate
        self.threshold = threshold

    def fit(self, x, y):
        x, y = self._pre_fit(x, y)
        coeffs = gd(x, y, self.rate, self.threshold)
        self.b, self.W = coeffs[0], coeffs[1:]


class LogisticRegression(GDLinearRegression):
    """Logistic regression model."""

    def __init__(self, learning_rate=0.05, threshold=1e-15):
        super().__init__(learning_rate, threshold)
    
    @staticmethod
    def __pred(x, w):
        return 1 / (1 + np.exp(- (x.dot(w))))

    def fit(self, X, y):
        X, y = self._pre_fit(X, y)
        coeffs = gd(X, y, self.rate, self.threshold, pred=self.__pred)
        self.b, self.W = coeffs[0], coeffs[1:]


    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        return np.where(Z > 0.5, 1, 0)
