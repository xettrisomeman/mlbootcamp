import numpy as np

def linear_regression(X, y, reg_value = None):
    
    bias = np.ones(X.shape[0])

    X = X.values
    
    X = np.column_stack([bias, X])

    # find transpose
    XTX = X.T.dot(X)
    
    if reg_value:
        XTX = XTX + reg_value * np.eye(XTX.shape[0])

    # inverse of transpose
    inv_XTX = np.linalg.inv(XTX)

    # find weight
    weight = inv_XTX.dot(X.T).dot(y)

    #return bias, weight
 
    return weight[0], weight[1:]


def get_training_mean(X):
    x_train_mean = X.reviews_per_month.mean()
    return round(x_train_mean, 3)


# fill the mean
def fill_training(X, types = None, mean = None):
    if types == "zero":
        return X.fillna(0)
    return X.fillna(mean)

# calculate the loss
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return round(np.sqrt(mse), 2)


def calculate_std(scores):
    return round(np.std(scores), 3)

