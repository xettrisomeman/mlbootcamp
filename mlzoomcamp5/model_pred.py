import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv("Churn.csv")


df.columns = df.columns.str.lower().str.replace(" ", "_")

categorical_columns = list(df.columns[df.dtypes == "object"])


for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")


df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == "yes").astype(int)

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [column for column in list(df.columns) if column not in numerical + ['churn', 'customerid']]


def train(df_train, y_train, C= 1.0):
    dicts = df_train[categorical + numerical].to_dict("records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C = C, max_iter = 1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df_val, dv, model):
    dicts = df_val[categorical + numerical].to_dict("records")

    X_val = dv.transform(dicts)

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred



C = 1.0
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):

    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
   
    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)


    auc = roc_auc_score(y_val, y_pred)

    scores.append(auc)

# print(f'C = {C}, {np.mean(scores):.3f}, {np.std(scores):.3f}')
# print(scores)


#do prediction
dv, model= train(df_full_train, df_full_train.churn.values, C = 1.0)

y_test = df_test.churn.values
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)

# print(auc)


# save the model using joblib
import joblib
def save_model(filename):
    joblib.dump((dv, model), filename)

save_model(f"Model_{C}.bin")



