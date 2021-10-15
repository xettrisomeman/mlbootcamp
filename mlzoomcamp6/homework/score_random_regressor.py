from train_test_splits import (
        X_train, X_val, y_train, y_val
        )


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt


n_estimators_params = list(range(10, 201, 10))


scores = []

def train(estimator):
    rf = RandomForestRegressor(n_estimators = estimator, random_state = 1)
    
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    # scores.append(rmse)
    scores.append((estimator, rmse))



for estimator in tqdm(n_estimators_params):
    train(estimator)



#plot the data
# plt.plot(n_estimators_params, scores, marker = ".")
# plt.show()

print(scores)
#answer -> 120 


