import pytest
from src.linear_regression import LinearRegression, GDLinearRegression, UnfittedModelException

def test_unfitted_model():
    model = LinearRegression()
    with pytest.raises(UnfittedModelException):
        model.predict(1)

def test_creating_gd_variant():
    with pytest.raises(ValueError):
        GDLinearRegression(learning_rate=-1)
    with pytest.raises(ValueError):
        GDLinearRegression(threshold=-1)
