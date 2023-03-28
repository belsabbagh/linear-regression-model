"""
NOTE: This function runs weirdly slow due to the fact that it is using a for loop to iterate over the data.
The last test took 5 minutes to run.
"""


from timeit import default_timer as dt
import numpy as np
import pandas as pd
from src.linear_regression.gradient_descent import mse
from src.plotter import Plotter
from src.linear_regression import (
    GDLinearRegression,
    LinearRegression,
    LogisticRegression,
)

df = pd.read_csv("data/data.csv")
X, y = df[["Height"]], df["Weight"]


def test_model(X, y, model, plot=True):
    model_name = model.__class__.__name__
    print(f"Testing {model_name} model...")
    if plot:
        Plotter.plot_points(X, y)
    s = dt()
    model.fit(X, y)
    print(f"Took {round(dt() - s, 5)}s to fit {model}.")
    pred = [model.predict(i) for i in X.values]
    if plot:
        Plotter.plot_line(X, pred)
    print(f"Mean squared error: {mse(y, pred)}\n-----------------------------------")
    Plotter.show()


test_model(X, y, LinearRegression())
test_model(X, y, GDLinearRegression(learning_rate=0.2))


def test_weather_history():
    df = pd.read_csv("data/weather-history.csv", dtype=np.float32)
    label = "Temperature (C)"
    X, y = df.loc[:, df.columns != label], df[label]
    X = (df - df.mean()) / df.std()  # (X-X.min())/(X.max()-X.min())
    test_model(X, y, GDLinearRegression(learning_rate=0.5, threshold=1e-4), plot=False)


def test_breast_cancer():
    df = pd.read_csv("data/breast-cancer.csv")
    label = "diagnosis"
    X, y = df.loc[:, df.columns != label], df[label]
    X = X.astype(np.float32)
    X = (df - df.mean()) / df.std()  # (X-X.min())/(X.max()-X.min())
    y = y.map({"M": 1, "B": 0})
    test_model(X, y, LogisticRegression(), plot=False)


test_weather_history()
