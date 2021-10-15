from train_test_splits import (
        X_train, X_val, y_train, y_val
        )
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

max_depth_range = np.arange(10, 30, 5)
n_estimators_paras = list(range(10, 201, 10))
# print(max_depth_range)


def train(estimator, max_depth):
    rf = RandomForestRegressor(max_depth = max_depth, n_estimators = estimator, random_state = 10)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse

for max_depth in tqdm(max_depth_range):
    rmse_scores = []
    for estimator in n_estimators_paras:
        rmse = train(estimator, max_depth)
        rmse_scores.append((max_depth, rmse))
    sorted_rmse = sorted(rmse_scores, key = lambda x: x[1])
    print(sorted_rmse)

# answer -> max_depth : 15 (random_state = 1)
# answer -> max_depth : 15 (random_state = 10) 

