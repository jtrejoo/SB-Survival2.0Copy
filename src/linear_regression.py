#linearregression.py
#got this from lecture slides:

import numpy as np

def ordinary_least_squares(X, y):
    # Add bias term
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # Closed form solution: β = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def predict(X, beta):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X @ beta
