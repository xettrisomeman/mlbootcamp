import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from split import split_the_data

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("AB_NYC_2019.csv")

# print(df.head())
# print(df.shape)


# FEATURES
features = [
    "neighbourhood_group",
    "room_type",
    'latitude',
    'longitude',
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    "calculated_host_listings_count",
    "availability_365"
]

df_data = df.loc[:, features]

# check for missing values
# print(df_data.isna().sum())

# -> reviews_per_month has 10052

# Fill the missing values with 0
df_data.fillna(0, inplace=True)

# print(df_data.isna().sum())



# Question 1: Find most frequent observation for the column -> neighbourhood_group

frequent_neightbourhodd_group = df_data['neighbourhood_group'].value_counts().index[0]
# print(frequent_neightbourhodd_group)
# Answer -> manhattan : 21661


# SPLIT THE DATA

X_train, X_test, X_val, y_train, y_test, y_val = split_the_data(df_data)
"""
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)
"""



# Question 2

# Create the (correlation matrix) for numerical feature of the dataset.

numerical = list(X_train.dtypes[X_train.dtypes != "object"].index)
# print(numerical)

# calculate correlation
X_correlation = X_train[numerical].corr().round(3)
# print(X_correlation)


# plot the correlation matrix with seaborn
"""
sns.heatmap(X_correlation, annot=True)
plt.savefig("correlation.jpg")
plt.show()
"""

# Answer -> reviews_per_month : number_of_reviews -> 0.590

# Make price binary

above_average = lambda x: 1 if x >= 152 else 0

def binary_price(value):
    return above_average(value)

y_train_binary = y_train.map(binary_price)
y_test_binary = y_test.map(binary_price)
y_val_binary = y_val.map(binary_price)


# Question 3
# Calculate mutual information

from sklearn.metrics import mutual_info_score

categorical = [x for x in list(X_train.columns) if x not in numerical]
# print(categorical)


neighbourhood_group_mutual_info = mutual_info_score(y_train_binary, X_train[categorical[0]])
# print(neighbourhood_group_mutual_info)

room_type_mutual_info = mutual_info_score(y_train_binary, X_train[categorical[1]])
# print(room_type_mutual_info)

# Answer -> room_type -> 0.143



# Question 5
# train a logistic regression model
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def model():
    clf = LogisticRegression(solver = "lbfgs", C=1.0, random_state = 42)
    return clf



def train(X_train, y_train_binary, X_val, y_val_binary):
    #turn the data into dictionary at first
    X_train_dic = X_train.to_dict("records")
    
    # print(X_train_dic)
    dv = DictVectorizer(sparse=False)

    one_hot_train = dv.fit_transform(X_train_dic)

    # print(one_hot)
    # print(dv.get_feature_names())

    # train a logistic regression model

    clf = model()
    clf.fit(one_hot_train, y_train_binary)


    #now predict the score on validation dataset

    X_val_dic = X_val.to_dict("records")
    one_hot_val = dv.transform(X_val_dic)

    y_pred_val = clf.predict(one_hot_val)
    val_accuracy = accuracy_score(y_val_binary, y_pred_val)
    return val_accuracy



val_accuracy = train(X_train, y_train_binary, X_val, y_val_binary)
# print(val_accuracy)
# Answer ->validation accuracy_score -> 0.79



# Question 5
# feature elemination technique

def calculate_accuracy_by_removing_features(clf,features, val_accuracy, X_train, y_train_binary, X_val, y_val_binary):
    accuracy_features = {}
    features.remove("price")
    for feature in list(X_train.columns):
        selected_feature = features.copy()
        selected_feature.remove(feature)
        # print(selected_feature)
        X_train_dict = X_train[selected_feature].to_dict("records")
        X_val_dict = X_val[selected_feature].to_dict("records")


        # fit and transform to dict
        dv = DictVectorizer()
        X_train_onehot = dv.fit_transform(X_train_dict)
        X_val_onehot = dv.transform(X_val_dict)

        # fit the model
        clf.fit(X_train_onehot, y_train_binary)
        y_val_pred = clf.predict(X_val_onehot)


        # calculate accuracy
        accuracy_without_feature = accuracy_score(y_val_binary, y_val_pred)

        # calculate difference
        difference = val_accuracy - accuracy_without_feature
        accuracy_features[f"Without {feature}"] = round(difference , 5)
        print(f"without {feature}")
        print(accuracy_without_feature)

    return accuracy_features

# accuracy_features = calculate_accuracy_by_removing_features(model(),features, val_accuracy, X_train, y_train_binary, X_val, y_val_binary)

# print(accuracy_features)

# Answer -> number_of_reviews = -0.00041



# Question 6
# use linear ridge regression model

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error



def apply_log(price):
    return np.log1p(price)

y_train_log = y_train.map(apply_log)
# y_test_log = y_test.map(apply_log)
y_val_log = y_val.map(apply_log)


# print(y_train_log, y_test_log, y_val_log)




def rmse(y_test, y_pred):
    se = (y_test - y_pred) ** 2
    mse = np.mean(se)
    return round(np.sqrt(mse), 3)

from sklearn.metrics import r2_score

def ridge_train(X_train, y_train_log, X_val, y_val_log, alpha_list):

    alpha_list = [0, 0.01, 0.1, 1, 10]
    rmse_alpha = {}
    X_train = X_train.to_dict("records")
    X_val = X_val.to_dict("records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(X_train)

    X_val = dv.transform(X_val)

    for alpha in alpha_list:
        reg = Ridge(alpha = alpha)
        reg.fit(X_train, y_train_log)

        y_pred = reg.predict(X_val)

        root_mean_squared_score = rmse(y_val_log, y_pred)
        rmse_alpha[f"Alpha: {alpha}"] = root_mean_squared_score


# ridge_train(X_train, y_train_log, X_val, y_val_log, alpha)
# print(rmse_alpha)

#Answer alpha : 0.0, 0.01 -> 0.497

