import numpy as np

from base_model import df_train, df_val, y_train, y_val
from base_model import base
from train import train_linear_regression
from rmse import rmse


import matplotlib.pyplot as plt
import seaborn as sns


def prepare_x(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year

    features.append('age')

    for v in [2, 3, 4]:
        df[f"num_doors_{v}"] = (df_train.number_of_doors == v).astype('int')
        features.append(f"num_doors_{v}")

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X

X_train = prepare_x(df_train)

# print(X_train)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_x(df_val)

y_pred = w0 + X_val @ w

print(rmse(y_val, y_pred))
