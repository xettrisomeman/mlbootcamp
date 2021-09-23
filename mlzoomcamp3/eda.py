from validation_framework import df_full_train


import seaborn as sns
import matplotlib.pyplot as plt

df_full_train = df_full_train.reset_index(drop=True)


# check for missing values
# print(df_full_train.isnull().sum())


# check churn distribution
# sns.countplot(x = df_full_train.churn)
# plt.savefig("x_y_countplot.jpg")
# plt.show()


# calculate churn rate
global_churn_rate = df_full_train.churn.mean()
# print(global_churn_rate)
# print(round(global_churn_rate, 2))

numerical = ["tenure", "monthlycharges", "totalcharges"]

categorical = [x for x in list(df_full_train.columns) if x not in numerical + ['customerid', 'churn']]

# print(categorical)

#check unique values (categorical)
# print(df_full_train[categorical].nunique())

