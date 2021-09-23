# Training logistic regresison with Scikit-Learn

from ohe import X_train, X_val
from validation_framework import y_train, y_val, df_val, df_train

import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


# set the model
model = LogisticRegression()

# fit the model
model.fit(X_train , y_train)

# print the co-efficients(weights)
# print(model.coef_[0].round(3))

#print the intercept - w0
# print(model.intercept_)


# soft prediction -> get prediction values of the class
# print(model.predict_proba(X_train))

# hard prediction -> get the predicted value (class with higher probability)
# print(model.predict(X_train))

# get soft prediction
y_pred = model.predict_proba(X_val)[:, 1] 


churn_decision = (y_pred >= 0.5)

# check how accurate our models are

# print((y_val == churn_decision).mean())

# print(y_val)
# print(churn_decision.astype(int))

df_pred = pd.DataFrame()
df_pred["probability"] = y_pred
df_pred["prediction"] = churn_decision.astype(int)
df_pred["actual"] = y_val

# print(df_pred)

df_pred["correct"] = y_val == churn_decision.astype(int)

# print(df_pred)
# print(df_pred.correct.mean())


# MODEL INTERPRETATION



# print(model.coef_[0].round(3))

small = ["contract", "tenure", "monthlycharges"]

# print(df_train[small])


dicts_train_small = df_train[small].to_dict("records")
dicts_val_small = df_val[small].to_dict("records")


dv_small = DictVectorizer(sparse = False)

dv_small.fit(dicts_train_small)

# print(dv_small.get_feature_names())




X_train_small = dv_small.transform(dicts_train_small)
X_val_small = dv_small.transform(dicts_val_small)

model_small = LogisticRegression()

model_small.fit(X_train_small, y_train)


w0 = model_small.intercept_[0]
w = model_small.coef_[0]


# print(dict(zip(dv_small.get_feature_names(), w.round(3))))


def logistic_regression(X_val_small, w0, w):
    y_pred = w0 + X_val_small.dot(w)
    sigmoid = 1/ (1+np.exp(-y_pred))
    decision_value = sigmoid >= 0.5
    return decision_value.astype(int)


y_val_pred = logistic_regression(X_val_small, w0, w)

# print(y_val_pred)

