import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    D = X.shape [1]
    N = X.shape [0]
    w = np.zeros (X.shape [1])
    b = float (0)

    for i in range (steps):
        y_hat = _sigmoid(X.dot(w)+b)
        e = y_hat - y
        dw = X.T.dot(e)/N
        db = np.sum(e)/N
        w = w -lr*dw
        b = b-lr*db

    return w,b
    
"""for mỗi step:
    tính z
    tính y_hat
    tính error
    tính gradient
    update w, b"""
        
    