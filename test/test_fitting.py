import pandas as pd
from src.linear_regression import fitting_functions as ff


def test_least_squares_fit():
    x, y = pd.Series([1, 2, 3]), pd.Series([2, 4, 6])
    m, c = ff.least_squares(x,y)
    assert m == 2
    assert c == 0
    

def test_gd_linear_regression():
    x, y = pd.Series([1, 2, 3]), pd.Series([2, 4, 6])
    m, c = ff.gd(x,y)
    assert round(m) == 2
    assert round(c) == 0
