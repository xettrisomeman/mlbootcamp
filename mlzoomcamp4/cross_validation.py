from data_model import categorical, numerical, df_train, df_val, y_train, y_val, df_full_train

import numpy as np


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import KFold

from tqdm import tqdm


def remove_churn(categorical, numerical):
    total = categorical+ numerical
    total.remove("churn")
    return total


def train(df, y, C = 1.0, max_iter = 100):
 
    total = remove_churn(categorical, numerical)

    dicts_df = df[total].to_dict("records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dicts_df)

    model = LogisticRegression(C= C, max_iter=max_iter)

    model.fit(X_train, y)

    return dv, model

dv, model = train(df_train, y_train)



def predict(df, dv, model):

    total = remove_churn(categorical, numerical)


    dicts = df[total].to_dict("records")

    X_val = dv.transform(dicts)
    

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


y_pred = predict(df_val, dv, model)
decision = (y_pred >= 0.5)

# print((decision == y_val).sum() / len(y_val))

# split 1/10th of dataset in every iteration ( it means splits dataset into 10 part with 1/10 in val side)
kfold = KFold(n_splits = 10, shuffle=True, random_state = 42)

# split method
# takes in feature matrix, returns iterator of indexes


train_idx, val_idx = next(kfold.split(df_full_train))
# print(len(train_idx))




def train_full(df_full_train, C, n_splits, max_iter):
    scores = []
    # split 1/10th of dataset in every iteration ( it means splits dataset into 10 part with 1/10 in val side)
    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = 42)

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_val = df_val.churn.values
        y_train = df_train.churn.values

        dv, model = train(df_train, y_train, C = C, max_iter = max_iter)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f"C= {C} {np.mean(scores):.3f} +- {np.std(scores):.3f}")

# c_params = [1e-2, 1e-1, 1e-0, 0.5, 1, 5, 10]

# n_splits = 5
# max_iter = 1000

# for C in tqdm(c_params):
#     train_full(df_full_train, C = C, n_splits = n_splits, max_iter = max_iter)


dv, model = train(df_full_train, df_full_train.churn, C= 1.0, max_iter = 1000)

y_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_pred)

print(round(auc,3))
#gives 0.85 auc
