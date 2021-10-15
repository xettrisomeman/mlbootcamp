from data_clean import (
        df_test, df_val,df_train,
        y_test, y_val, y_train
        )

import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt



X_dicts = df_train.to_dict("records")
X_val_dicts = df_val.to_dict("records")

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(X_dicts)
X_val = dv.transform(X_val_dicts)


max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]

def calculate_tree(max_depths):
    for d in max_depths:
        dt = DecisionTreeClassifier(max_depth = d)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]

        print(f"{d}=> {roc_auc_score(y_val, y_pred):.3f}")


# calculate_tree(max_depths)



scores = []
for d in [4, 5, 6]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth = d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]

        scores.append([d, s, roc_auc_score(y_val, y_pred)])

# print(scores)


df_scores = pd.DataFrame(scores, columns = ["max_depth", "min_samples_leaf", "auc_score"])
# print(df_scores)

# print(df_scores.sort_values(by="auc_score", ascending=False))


df_scores_pivot = df_scores.pivot(index = "min_samples_leaf", columns = ["max_depth"], values=["auc_score"]).round(3)

#visualize as heatmap

# sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
# plt.savefig("heatmap_pivot.jpg")
# plt.show()




scores = []
for d in [4, 5, 6, 7, 10, 15, 20, None]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth = d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]

        scores.append([d, s, roc_auc_score(y_val, y_pred)])

# print(scores)


df_scores = pd.DataFrame(scores, columns = ["max_depth", "min_samples_leaf", "auc_score"])
# print(df_scores)

# print(df_scores.sort_values(by="auc_score", ascending=False))


df_scores_pivot = df_scores.pivot(index = "min_samples_leaf", columns = ["max_depth"], values=["auc_score"]).round(3)

#visualize as heatmap

# sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")
# plt.savefig("heatmap_pivot2.jpg")
# plt.show()



