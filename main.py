"""
NOTE: This function runs weirdly slow due to the fact that it is using a for loop to iterate over the data.
The last test took 5 minutes to run.
"""


from datetime import timedelta
from timeit import default_timer
import pandas as pd
from src import split_dataset
from src.linear_regression.fitting_functions import mse
from src.plotter import Plotter
from src.linear_regression import GDLinearRegression, LinearRegression

def test_model(model):
    df = pd.read_csv('data/data.csv')
    label = 'Weight'
    split_at = 11
    Plotter.plot_points(df['Height'], df[label])
    train, test = split_dataset(df, split_at)
    start = default_timer()
    model.fit(train['Height'], train[label])
    print(f"Time taken to fit {model.__class__.__name__}: {timedelta(seconds=default_timer() - start)}.")
    Plotter.plot_line(df['Height'], [model.predict(i) for i in df['Height']])
    pred = [model.predict(i) for i in test['Height']]
    print(f"Mean squared error: {mse(test[label], pred)}")
    Plotter.show()


test_model(GDLinearRegression(learning_rate=0.1, threshold=1e-9))
test_model(LinearRegression())
