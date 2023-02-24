import pytest
from src.linear_regression import LinearRegression, UnfittedModelException

def test_unfitted_model():
    model = LinearRegression()
    with pytest.raises(UnfittedModelException):
        model.predict(1)
