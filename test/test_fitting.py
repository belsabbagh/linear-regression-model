import numpy as np
import pandas as pd
from src.linear_regression import fitting_functions as ff

x1d = pd.DataFrame({
    'C1': [1, 2, 3, 4],
})
x2d = pd.DataFrame({
    'C1': [1, 2, 3, 4],
    'C2': [1.5, 2.5, 3.5, 4.5],
})
y = [2, 4, 6, 8]


def base_test_1d(x, y, fn):
    b = fn(x, y)
    res = [sum([k*p for k, p in zip(v, b[1:])]) + b[0]
           for _, v in x.iterrows()]
    assert [round(i, 3) for i in res] == y


def base_test_2d(fn):
    x = x2d
    b = fn(x, y)
    res = [sum([k*p for k, p in zip(v, b[1:])]) + b[0]
           for _, v in x.iterrows()]
    assert [round(i, 3) for i in res] == y


def test_1d_least_squares_fit():
    x = np.array(x1d)
    b = ff.least_squares(x, y)
    for i,j in zip(x,y):
        m = np.sum([k*p for k, p in zip([i], b[1:])])
        assert round(m + b[0]) == j 


def test_1d_gd_fit():
    x = np.array(x1d)
    b = ff.gd(x, y, rate=0.05, threshold=1e-15)
    for i,j in zip(x,y):
        m = np.sum([k*p for k, p in zip([i], b[1:])])
        assert round(m + b[0]) == j 


def test_2d_least_squares_fit():
    base_test_2d(ff.least_squares)


def test_2d_gd_fit():
    base_test_2d(ff.gd)
