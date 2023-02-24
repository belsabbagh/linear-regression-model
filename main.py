import pandas as pd
from src import Plotter, split_dataset
from src.linear_regression import LinearRegression

def test_model():
    df = pd.read_csv('data/data.csv')
    label = 'Weight'
    split_at = 11
    Plotter.plot_points(df['Height'], df[label])
    train, test = split_dataset(df, split_at)
    model = LinearRegression()
    # model.fit_gd(train['Height'], train[label], 0.01, 1e-8)
    model.fit(train['Height'], train[label])
    Plotter.plot_line(df['Height'], [model.predict(i) for i in df['Height']])
    res = [model.predict(i) for i in test['Height']]
    print(f"Mean squared error: {LinearRegression.mse(test[label], res)}")
    Plotter.show()
    print(res)


test_model()
