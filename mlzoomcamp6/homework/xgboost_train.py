from train_test_splits import (
        X_train, X_val, y_train, y_val
        )
import xgboost as xgb
from sklearn.metrics import roc_auc_score



dtrain = xgb.DMatrix(X_train, label = y_train)
dval = xgb.DMatrix(X_val, label = y_val)

# watchlist
watchlist = [(dtrain, "train"), (dval, "eval")]

# set params
def train_model(eta, dtrain, dval, y_val):
    xgb_params = {
        'eta': eta, 
        'max_depth': 6,
        'min_child_weight': 1,

        'objective': 'reg:squarederror',
        'nthread': 8,

        'seed': 1,
        'verbosity': 1,
    }


    model = xgb.train(params = xgb_params,
            dtrain = dtrain,
            num_boost_round = 100,
            verbose_eval = 100,
            evals = watchlist)

for eta in [0.3, 0.1, 0.01]:
    print(f"eta {eta}")
    train_model(eta, dtrain, dval, y_val)


#answer -> 0.1
