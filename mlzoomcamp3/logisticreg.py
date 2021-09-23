import numpy as np
import matplotlib.pyplot as plt


# from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    return 1/(1 + np.exp(-z))

z = np.linspace(-7,5, 51)

w , w0 = [x for x in range(X.shape)], 0

def logistic_regression(xi):

    score = w0

    for j in range(len(w)):
        score = score + xi[j] * w[j]
    
    result = sigmoid(score)
    return result


# print(sigmoid(z))

# plot sigmoid

# plt.plot(z, sigmoid(z))
# plt.show()

