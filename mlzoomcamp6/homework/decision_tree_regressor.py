
import pandas as pd

from train_test_splits import (
        X_train, y_train, dv
        )
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.feature_extraction import DictVectorizer


dtr = DecisionTreeRegressor(max_depth = 1)

dtr.fit(X_train, y_train)


print(export_text(dtr, feature_names = dv.get_feature_names()))
#answer -> room_type










