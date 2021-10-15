from data_clean import (df_train, df_val, df_test,
        y_train, y_test, y_val)




def assess_risk(client):
    if client["records"] == "yes":
        if client["job"] == "parttime":
            return "default"
        else:
            return "ok"
    else:
        if client["assets"] > 6000:
            return "ok"
        else:
            return "default"
        

xi = df_train.iloc[0].to_dict()

# print(assess_risk(xi))

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

train_dicts = df_train.to_dict("records")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)

# dt = DecisionTreeClassifier()
dt = DecisionTreeClassifier(max_depth = 2)

dt.fit(X_train, y_train)

y_pred_train = dt.predict_proba(X_train)[:, 1]

print(roc_auc_score(y_train, y_pred_train))


val_dicts = df_val.to_dict("records")

X_val = dv.transform(val_dicts)

y_pred = dt.predict_proba(X_val)[:, 1]

print(roc_auc_score(y_val, y_pred))

print(export_text(dt, feature_names = dv.get_feature_names()))

# print(df_train.head())



