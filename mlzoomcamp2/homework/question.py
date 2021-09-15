import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from linear_reg import linear_regression, get_training_mean, fill_training, rmse, calculate_std
from split import split_the_data

# read the dataset

df = pd.read_csv("AB_NYC_2019.csv")


# extract the features (choose the features)
features = [
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    "calculated_host_listings_count",
    "availability_365"
]

df_data = df[features]


# find feature with missing values

def get_missing_values(df):
    return df.isna().sum()

# Q.1 
# -> feature (reviews_per_month) has 10052 missing values.
# print(get_missing_values(df_data))



# find  median for variable 'minimum_nights'?

def find_median(df_feature):
    feature = df_data[df_feature]
    # arange the data set in ascending order
    sorted_feature = np.sort(feature)

    # get the length of the dataset
    N = sorted_feature.shape[0]

    percentile = 50

    # find the (50% percentile) median
    
    n = round((percentile / 100) * N)

    return sorted_feature[n]
    

# Q.2 
# -> 3
# print(find_median('minimum_nights'))


# split the data
X = df_data.drop("price", axis=1)
y = np.log1p(df_data['price'])

X_train, X_val, X_test, y_train, y_val, y_test = split_the_data(X, y, 60, 20, 20, seed=42)


# print(X_train, X_val.shape, X_test.shape)



# Question 3.
# Deal with missing values

# get the mean

X_train_mean = get_training_mean(X_train)
X_train_zero = fill_training(X_train, types="zero")

# get weight and bias
bias , weight = linear_regression(X_train_zero, y_train)

# predict
y_pred = bias + X_train.dot(weight)



# print(rmse(y_train, y_pred))

# sns.histplot(y_pred, color="red", alpha=0.5)
# sns.histplot(y_train, color="blue", alpha=0.5)
# plt.savefig("y_pred_true.png")
# plt.show()

# validation data evaluation



X_val_zero = fill_training(X_val, types="zero")
X_val_mean = fill_training(X_val, mean=X_train_mean)



# compare rmse
y_val_zero = bias + X_val_zero.dot(weight)

loss_zero = rmse(y_val, y_val_zero)

y_val_mean = bias + X_val_mean.dot(weight)

loss_mean = rmse(y_val, y_val_mean)

# print(loss_zero, loss_mean)
# answer -> no difference , both are equal. 0.64



# Question 4:
reg_value = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1]
rmse_loss_values = {}
for reg in reg_value:
    bias, weight = linear_regression(X_train_zero, y_train, reg_value=reg)
    X_val_zero = fill_training(X_val, types="zero")
    y_val_pred = bias + X_val_zero.dot(weight)
    rmsee = rmse(y_val, y_val_pred)
    rmse_loss_values[reg] = rmsee

# Answer -> r: 0, 0.0000001, 0.0001, 0.001-> 0.64 rmse
# print(rmse_loss_values)


# Question 5

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

rmse_scores = {}
for seed in seeds:
    X_train, X_val, X_test, y_train, y_val, y_test = split_the_data(X, y, 60, 20, 20, seed=seed)

    # fill with zero
    X_train = fill_training(X_train, types="zero")
    X_val = fill_training(X_val, types="zero")

    # train the model
    bias, weight = linear_regression(X_train, y_train)

    y_pred_val = bias + X_val.dot(weight)

    rmse_score = rmse(y_val, y_pred_val)

    rmse_scores[f"Seed {seed}"] = rmse_score

std_rmse = calculate_std(list(rmse_scores.values()))

# answer -> 0.008
# print(std_rmse)


# Question 6


# use seed 9
seed = 9
X_train, X_val, X_test, y_train, y_val, y_test = split_the_data(X, y, 60, 20, 20, seed=seed)

# combine train and validation datasets

X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])

# fill with zero
X_train = fill_training(X_train, types = "zero")

bias, weight = linear_regression(X_train, y_train, reg_value=0.001)

y_pred_test = bias + X_test.dot(weight)

rmse = rmse(y_test, y_pred_test)

# print(rmse)
# Answer-> 0.62
