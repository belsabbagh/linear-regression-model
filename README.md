# Linear Regression Model

This is a simple linear regression model written in Python. It implements gradient descent and normal equation to find the optimal parameters for the model.

## Usage

To use the model, simply import the model and create an instance of it. Then, call the fit method to train the model (or use fit_gd for gradient descent). Finally, call the predict method to make predictions.

```python
from linear_regression import LinearRegression

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```
