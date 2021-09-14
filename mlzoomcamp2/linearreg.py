import numpy as np

xi = [453, 11, 86]
w0 = 7.17
w = [0.01, 0.04, 0.002]


# def dot(xi, w):
#     n = len(xi)
#     res = 0.0
#     for j in range(n):
#         res += xi[j] * w[j]
#     return res

w_new = [w0] + w


# def linear_regression(xi):
#     xi = [1] + xi
#     return dot(xi, w_new)

# print(linear_regression(xi))


x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X)
# print(X)

def linear_regression(X):
    return X.dot(w_new)

print(linear_regression(X))
