import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("Churn.csv")


df.columns = df.columns.str.lower().str.replace(' ', "_")

df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)
categorical_columns = list(df.columns[df.dtypes == "object"])


for c in categorical_columns:
    if c != "churn":
        df[c] = df[c].str.lower().str.replace(" ", "_")


df.churn = df.churn.apply(lambda x: 1 if x == "Yes" else 0)

numerical = ["tenure", "monthlycharges", "totalcharges"]
categorical = [x for x in list(df.columns) if x not in numerical + ["customerid"]]


def get_clean_dataframe(dataframe, total):
    X_value = dataframe[total].reset_index(drop=True)
    y_value = dataframe["churn"].reset_index(drop=True).values
    return X_value, y_value

total = numerical + categorical
total.remove("churn")
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)

# print(df_train.columns)

df_train, y_train = get_clean_dataframe(df_train, total)
df_val, y_val = get_clean_dataframe(df_val, total)
df_test, y_test = get_clean_dataframe(df_test, total)
# print(df_train.shape, y_train.shape)
# print(df_val.shape, y_val.shape)
# print(df_test.shape, y_test.shape)
# print(df_train)

# change to dict
df_train_dict = df_train.to_dict('records')
# print(df_train_dic[0])
df_val_dict = df_val.to_dict("records")
# print(df_val_dict[0])
df_test_dict = df_test.to_dict("records")

# transform model
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(df_train_dict)
X_test = dv.transform(df_test_dict)
X_val = dv.transform(df_val_dict)

# print(X_train.shape, X_test.shape, X_val.shape)


# train model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:,1]
churn_decision = (y_pred) > 0.5
mean_churn = (y_val == churn_decision).mean()

# print(mean_churn)