from data_clean import (
        df_train, df_val, y_train, y_val
        )
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


import xgboost as xgb



X_train_dict = df_train.to_dict("records")
X_val_dict = df_val.to_dict("records")


dv = DictVectorizer(sparse=False)


X_train = dv.fit_transform(X_train_dict)
X_val = dv.transform(X_val_dict)


features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train , label = y_train, feature_names = features)
dval = xgb.DMatrix(X_val, label = y_val, feature_names = features)


xgb_params = {
        "eta" : 0.3,
        "max_depth": 6,
        "min_child_weight": 1,

        "objective": "binary:logistic",
        "nthread": 8,

        "seed": 1,
        "verbosity": 1,
}

# model  = xgb.train(xgb_params, dtrain, num_boost_round = 10)
# y_pred = model.predict(dval)
# auc = roc_auc_score(y_val, y_pred)
# print(auc)



# create watchlist to evaluate our model

watchlist = [(dtrain, "train"), (dval, "val")]

xgb_params = {
        "eta" : 0.3,
        "max_depth": 6,                 
        "min_child_weight": 1,

        "objective": "binary:logistic",
        'eval_metric': "auc",
        "nthread": 8,

        "seed": 1,
        "verbosity": 1,
}

model = xgb.train(xgb_params, dtrain, evals = watchlist, num_boost_round =200, verbose_eval = 5)
#answer -> 0.1


