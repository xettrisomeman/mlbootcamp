from train_test_splits import (
        X_train, X_val, y_train, y_val
        )
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import numpy as np




rf = RandomForestRegressor(n_estimators = 10, random_state = 1, n_jobs = -1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)


print(round(np.sqrt(mean_squared_error(y_val, y_pred)), 4))
# answer = 0.459



