from price_prediction import *
from train import train_linear_regression


import matplotlib.pyplot as plt
import seaborn as sns

# print(df_train.columns)


base = ["engine_hp", "engine_cylinders", "highway_mpg",
        "city_mpg", "popularity"]



X_train = df_train[base].fillna(0).values

# print(X_train)
w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

# sns.histplot(y_pred, color='g', bins=50)
# sns.histplot(y_train, color='blue', bins=50)
# plt.show()
