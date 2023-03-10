"""
NOTE: This function runs weirdly slow due to the fact that it is using a for loop to iterate over the data.
The last test took 5 minutes to run.
"""


from timeit import default_timer as dt
import pandas as pd
from src.linear_regression.fitting_functions import mse
from src.plotter import Plotter
from src.linear_regression import GDLinearRegression, LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('data/data.csv')
X, y = df[['Height']], df['Weight']


def test_model(X, y, model, plot=True):
    if plot:
        Plotter.plot_points(X, y)
    s = dt()
    model.fit(X, y)
    print(f"Took {round(dt() - s, 5)}s to fit {model.__class__.__name__}.")
    pred = [model.predict(i) for i in X.values]
    if plot:
        Plotter.plot_line(X, pred)
    print(f"Mean squared error: {mse(y, pred)}")
    Plotter.show()


# test_model(X, y, LinearRegression())
# test_model(X, y, GDLinearRegression())

df = pd.read_csv('data/iris.csv')
df['Species'] = LabelEncoder().fit_transform(df['Species'])
X, y = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']], df['Species']
test_model(X, y, LogisticRegression(learning_rate=0.01), plot=False)
