#extract feature importance

from train_test_splits import (
        X_train, y_train, X_val, y_val, dv
)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


clf = RandomForestRegressor(n_estimators = 10, max_depth = 20, random_state = 1, n_jobs = -1)


clf.fit(X_train, y_train)


datas = zip(clf.feature_importances_, dv.get_feature_names())


feature_importance = pd.DataFrame(datas, columns = ["feature_importance", "features"])


pivots = feature_importance.sort_values(by = "feature_importance")

# print(pivots)
# answer -> feature_importance : room_type = Entire home/apt


