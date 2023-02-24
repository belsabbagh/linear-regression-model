import numpy as np
import pandas as pd
from src.linear_regression import GDLinearRegression, LinearRegression


def test_linear_regression():
    model = LinearRegression()
    x, y = pd.Series([1, 2, 3]), pd.Series([2, 4, 6])
    model.fit(x,y)
    for i, j in zip(x, y):
        assert model.predict(i) == j
    

def test_gd_linear_regression():
    model = GDLinearRegression()
    x, y = pd.Series([1, 2, 3]), pd.Series([2, 4, 6])
    model.fit(x,y)
    for i, j in zip(x, y):
        assert round(model.predict(i)) == j
