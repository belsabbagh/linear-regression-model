import numpy as np
import pandas as pd
from src.linear_regression import fitting_functions as ff

x1d = pd.DataFrame({
    'C1': [1, 2, 3, 4],
})
x2d = pd.DataFrame({
    'C1': [1, 2, 3, 4],
    'C2': [1, 2, 3, 4],
})
y = [2, 4, 6, 8]


def base_test_1d(x, y, fn):
    b = fn(x, y)
    res = [sum([k*p for k, p in zip(v, b[1:])]) + b[0]
           for _, v in x.iterrows()]
    assert res == y


def base_test_2d(fn):
    x = x2d
    b = fn(x, y)
    res = [sum([k*p for k, p in zip(v, b[1:])]) + b[0]
           for _, v in x.iterrows()]
    assert res == y


def test_1d_least_squares_fit():
    base_test_1d(x1d, y, ff.least_squares)


def test_1d_gd_fit():
    x = np.array([1, 2, 3, 4])
    b = ff.gd(x, y)
    for i,j in zip(x,y):
        assert round(sum([k*p for k, p in zip([i], b[1:])]) + b[0]) == j 


def test_2d_least_squares_fit():
    base_test_2d(ff.least_squares)


def test_2d_gd_fit():
    base_test_2d(ff.nd_gd)
