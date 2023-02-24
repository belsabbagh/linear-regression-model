import pandas as pd
from src.linear_regression import fitting_functions as ff


def base_test_1d(fn):
    x = pd.Series([1, 2, 3, 4])
    y = 2 * x
    m, c = fn(x, y)
    assert round(m) == 2
    assert round(c) == 0


def base_test_2d(fn):
    x = pd.Series([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = pd.Series([2, 4, 6, 8])
    m, c = fn(x, y)
    assert round(m) == 2
    assert round(c) == 0


def test_1d_least_squares_fit():
    base_test_1d(ff.least_squares)


def test_1d_gd_linear_regression():
    base_test_1d(ff.gd)


def test_2d_least_squares_fit():
    base_test_2d(ff.least_squares)


def test_2d_gd_linear_regression():
    base_test_2d(ff.gd)
