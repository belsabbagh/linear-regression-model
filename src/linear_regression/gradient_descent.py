import numpy as np


def add_constant(x: np.ndarray):
    return np.append(np.ones((len(x), 1)), x, axis=1)


def found_min(cost, prev_cost, threshold, max_iter, i):
    diff = abs(prev_cost-cost) if prev_cost is not None else 0
    return diff < threshold and diff != 0 or (max_iter is not None and i >= max_iter)


def mse(y_pred, y):
    return sum([(i-j)**2 for i, j in zip(y, y_pred)]) / len(y)


def derivative(n, x, y, y_predicted):
    """
    Derivative of the mean squared error cost function.
    The derivative of the MSE cost function is the negative of the average of the product of the input and the error.
    """
    return  -(1/n) * sum(x * (y-y_predicted))


def update_weights(rate, n, X, y, y_pred, weights):
    """Update weights using the derivative of the cost function."""
    return [w - rate * derivative(n, X.T[i], y, y_pred) for i, w in enumerate(weights)]


def gradient_descent(X: np.ndarray, y, rate=0.05, threshold=1e-15, max_iter=None, pred=None, verbose=False):
    pred = pred or (lambda x, w: x.dot(w))
    X = add_constant(X)
    n, p = X.shape
    weights, i, prev_cost, cost = np.zeros(p), 1, None, None
    while not found_min(cost, prev_cost, threshold, max_iter, i):
        prev_cost = cost if cost is not None else 0
        y_pred = pred(X, weights)
        cost = mse(y_pred, y)
        weights = update_weights(rate, n, X, y, y_pred, weights)
        if verbose:
            print(f'{i}: {dict(cost=cost, weights=weights)}')
        i += 1
    print(f"Gradient Descent: It took {i} iterations to converge.")
    return weights
