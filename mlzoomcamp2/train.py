import numpy as np


X = [
    [148, 24, 1385], 
    [132, 25, 2031], 
    [453, 11, 86], 
    [158, 24, 185], 
    [172, 25, 201], 
    [413, 11, 86], 
    [38, 54, 185], 
    [142, 25, 431],
    [453, 31, 86]
]

X = np.array(X)
# print(X)


# add bias term
bias = np.ones(X.shape[0])
# X = np.column_stack([bias, X])
# print(X)


y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]
y = np.array(y)


XTX = X.T @ X
# print(XTX)

XTX_inv = np.linalg.inv(XTX)
# print(XTX_inv)

# formula : X.T * X^-1 * Y
W_full = XTX_inv.dot(X.T).dot(y)

w0 = W_full[0]
w = W_full[1:]

# print(w0, w)


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

w0, w = train_linear_regression(X, y)
# print(w0, w)
