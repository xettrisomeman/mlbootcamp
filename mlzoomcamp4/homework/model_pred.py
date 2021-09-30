from prepare_data import df as df_full_train

import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


from sklearn.model_selection import KFold

# what are the categorical variables ? what are the numerical?

categorical = list(df_full_train.columns[df_full_train.dtypes  == "object"])
numerical = list(df_full_train.columns[df_full_train.dtypes != "object"])
# remove target variable
numerical.remove("default")
# print(categorical, numerical)


# split the data
df_train, df_test = train_test_split(df_full_train, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_train, test_size = 0.25, random_state = 1)

# print(df_train.shape, df_val.shape, df_test.shape)




# QUESTION 1-> COMPUTE ROC AUC
def compute_auc(df, numerical_columns):

    auc_scores = []
    
    target = df['default'].values

    for feature in numerical_columns:

        auc_score = roc_auc_score(target, df[feature])


        if auc_score < 0.5:
            auc_score = roc_auc_score(target, -df[feature])

        auc_scores.append((feature, auc_score))

    return sorted(auc_scores, key = lambda x: x[1], reverse = True)
"""
uncomment below line and run
"""
# auc_scores = compute_auc(df_train, numerical)
# print(auc_scores)
# ANSWER 1 -> seniortiy : 0.709


# QUESTION 2 -> find auc score of the model on validation dataset

# Training the model
# choose features
features = ['seniority', 'income', 'assets', 'records', 'job', 'home']



X_train = df_train[features]
y_train = df_train.default.reset_index(drop=True).values

X_test = df_test[features]
y_test = df_test.default.reset_index(drop = True).values

X_val = df_val[features]
y_val = df_val.default.reset_index(drop = True).values

# one-hot encoding
def one_hot_encoding(dataframe, train, dv):
    if train:
        dict_transform = dv.fit_transform(dataframe)
    dict_transform = dv.transform(dataframe)

    return dict_transform


def train(X_train, y_train, solver, C, max_iter):
    X_encoding = X_train.to_dict("records")

    dv = DictVectorizer(sparse = False)

    X_train = one_hot_encoding(X_encoding, train = True, dv=dv)

    model = LogisticRegression(solver= solver, C = C, max_iter = max_iter)

    model.fit(X_train, y_train)

    return dv, model

# dv, model = train(X_train, y_train, solver= "liblinear", C=1.0, max_iter=1000)


def predict(df, dv, model):
    df = df.to_dict("records")

    X_df = one_hot_encoding(df, train = False, dv= dv)

    y_pred = model.predict_proba(X_df)[:, 1]
    # y_pred = model.predict(X_df)

    return y_pred


# y_pred = predict(X_val, dv, model)
# print(round(roc_auc_score(y_val, y_pred), 3))
# Answer -> auc score: 0.811




# Question 3 -> compute precision and recall
def compute_precision_and_recall(y_val, y_pred):
    threshold = np.linspace(0, 1, 101)
    precision_scores = []
    recall_scores = []
    
    for t in threshold:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predicted_positive = (y_pred >= t)
        predicted_negative = (y_pred < t)

        tp = (actual_positive & predicted_positive).sum()
        fp = (actual_negative & predicted_positive).sum()

        tn = (actual_negative & predicted_negative).sum()
        fn = (actual_positive & predicted_negative).sum()

        precision = round(tp / (tp + fp), 3)

        recall = round(tp / (tp + fn), 3)

        precision_scores.append(precision)
        recall_scores.append(recall)

    return threshold, precision_scores, recall_scores

"""
uncomment below line to get answer
"""
# threshold, precision_scores, recall_scores = compute_precision_and_recall(y_val, y_pred)

# plt.figure(figsize=  (9, 6))
# plt.plot(threshold, precision_scores, label="precision_scores", marker='.')
# plt.plot(threshold, recall_scores, label = "recall_scores", marker='.')
# plt.xlabel("threshold")
# plt.legend()
# plt.savefig("precision_recall_plot.jpg")
# plt.show()

# Answer -> line_meet_point: 0.4




# Question 4: calculate F1-score
def calculate_f1_score(threshold, precision_scores, recall_scores):
    f1_score = []
    for index, t in enumerate(threshold):
        precision = precision_scores[index]
        recall = recall_scores[index]
        f1_score.append((t, round(2 * precision * recall / (precision + recall), 3)))
    return sorted(f1_score, key= lambda x: x[1], reverse=True)

"""
uncomment below line to get answer
"""
# f1_score = calculate_f1_score(threshold, precision_scores, recall_scores)
# print(f1_score)
# Answer -> threshold:0.32, f1_Score : 0.626



# Question 5: Using kfold cross validation

def train_with_kfold(dataframe, splits, features, solver, C, max_iter):
    kfold = KFold(n_splits=splits, shuffle=True, random_state = 1)
    
    auc_scores = []

    for train_idx, val_idx in kfold.split(dataframe):
        df_train = dataframe.iloc[train_idx]
        df_val = dataframe.iloc[val_idx]

        X_train = df_train[features]
        X_val = df_val[features]

        y_train = df_train.default.values
        y_val = df_val.default.values

        # training

        dv, model = train(X_train, y_train, solver=solver, C=C, max_iter=max_iter)

        #prediction
        y_pred = predict(X_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)

        auc_scores.append(auc)

    auc_mean = np.mean(auc_scores).round(3)
    auc_std = np.std(auc_scores).round(3)

    return auc_scores, auc_mean, auc_std

n_splits = 5
# auc_scores , mean, standard_deviation = train_with_kfold(df_full_train, splits = n_splits, features = features, solver = "liblinear", C = 1.0, max_iter = 1000)

# print(auc_scores)
# print(mean)
# print(standard_deviation)
# Answer ->  auc_std - > 0.021


# Question 5: using 5 fold cross validation for c


c_values = [0.01, 0.1, 1, 10]

mean_scores = []

for c in c_values:
    _, auc_mean, _= train_with_kfold(df_full_train, splits=5, features = features, solver = "liblinear", C = c, max_iter = 1000)
    mean_scores.append((c, auc_mean))

# print(sorted(mean_scores, key= lambda x: x[1], reverse=True))
# Answer -> 0.1 : 0.805





