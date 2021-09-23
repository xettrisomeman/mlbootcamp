from eda import df_full_train
from eda import categorical

from sklearn.metrics import mutual_info_score


contract = mutual_info_score(df_full_train.churn, df_full_train.contract)
# print(contract)

gender = mutual_info_score(df_full_train.churn, df_full_train.gender)
# print(gender)


def mutual_churn_score(series):
    return mutual_info_score(df_full_train.churn, series)

series = df_full_train[categorical].apply(mutual_churn_score)

# print(series.sort_values(ascending=False))
