from data_clean import (
        df_train, y_train, df_val, y_val
        )

import pandas as pd


from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt



X_dicts = df_train.to_dict("records")
X_val_dicts = df_val.to_dict("records")


dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(X_dicts)
X_val = dv.transform(X_val_dicts)



#rf = RandomForestClassifier(n_estimators = 10, random_state = 1)
# rf.fit(X_train, y_train)

# y_pred = rf.predict_proba(X_val)[:, 1]
# print(roc_auc_score(y_val, y_pred))




def train_randomforest(max_depth = None):
    scores = []
    for n in tqdm(range(10, 201, 10)):
        rf = RandomForestClassifier(n_estimators = n, max_depth = max_depth, random_state = 1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)


        scores.append((n, auc))
    return scores


# scores = train_randomforest()
# print(scores)
# df_scores = pd.DataFrame(scores, columns = ["n_estimators", "auc"])
# print(df_scores)


# plt.plot(df_scores.n_estimators, df_scores.auc)
# plt.savefig("scores_estimators.png")
# plt.show()

def train_randomforest():
    scores = []
    for d in [5, 10, 15]:
        for n in tqdm(range(10, 201, 10)):
            rf = RandomForestClassifier(n_estimators = n, max_depth = d, random_state = 1)# columns = ["max_depth", "n_estimators", "auc"]
            rf.fit(X_train, y_train)
                                                                                                  # df_scores = pd.DataFrame(scores, columns = columns)
            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)                                                    # print(df_scores)

            scores.append((d, n, auc))
    return scores

# scores = train_randomforest()
# print(scores)
# columns = ["max_depth", "estimators", "auc"]
# df_scores = pd.DataFrame(scores, columns = columns)


# print(df_scores)

# for d in  [5, 10, 15]:
#     fig, ax = plt.subplots()
#     df_subset = df_scores[df_scores.max_depth == d]
#     ax.plot(df_subset.estimators, df_subset.auc, label = f"max_depth={d}")
# plt.legend()
# plt.show()




max_depth = 10
def train_randomforest():
    scores = []
    for s in [1, 3, 5, 10, 50]:
        for n in tqdm(range(10, 201, 10)):
            rf = RandomForestClassifier(n_estimators = n, max_depth = max_depth,
                    min_samples_leaf = s, random_state = 1)# columns = ["max_depth", "n_estimators", "auc"]
            rf.fit(X_train, y_train)
                                                                                                  # df_scores = pd.DataFrame(scores, columns = columns)
            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)                                                    # print(df_scores)

            scores.append((s, n, auc))
    return scores

scores = train_randomforest()
# print(scores)
columns = ["min_samples_leaf", "n_estimators", "auc"]
df_scores = pd.DataFrame(scores, columns = columns)


# print(df_scores)

for s in  [1, 3, 5, 10, 50]:
    fig, ax = plt.subplots()
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    ax.plot(df_subset.n_estimators, df_subset.auc, label = f"min_samples_leaf={s}")
    plt.legend()
plt.show()



