# Root mean squared error
import numpy as np
from price_prediction import *
from base_model import y_train, y_pred, base
from train import train_linear_regression

## -> sqrt(mean(sum((y_pred - y)**2)))

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# print(rmse(y_train, y_pred))

def prepare_x(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_x(df_train)
w0, w = train_linear_regression(X_train, y_train)

# print(w0, w)

X_val = prepare_x(df_val)

y_pred = w0 + X_val.dot(w)


# print the error
# print(rmse(y_val, y_pred))

