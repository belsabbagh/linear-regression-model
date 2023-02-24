from timeit import default_timer
import pandas as pd
from src import split_dataset
from src.linear_regression.fitting_functions import mse
from src.plotter import Plotter
from src.linear_regression import LinearRegression, GDLinearRegression

def test_model(model):
    df = pd.read_csv('data/data.csv')
    label = 'Weight'
    split_at = 11
    Plotter.plot_points(df['Height'], df[label])
    train, test = split_dataset(df, split_at)
    start = default_timer()
    model.fit(train['Height'], train[label])
    print(f"Time taken to fit {model.__class__.__name__}: {round(default_timer() - start, 2)} seconds.")
    Plotter.plot_line(df['Height'], [model.predict(i) for i in df['Height']])
    res = [model.predict(i) for i in test['Height']]
    print(f"Mean squared error: {mse(test[label], res)}")
    Plotter.show()


test_model(LinearRegression())
test_model(GDLinearRegression())
