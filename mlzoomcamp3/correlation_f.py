from eda import df_full_train, numerical


# get max value of (column) tenure
# print(df_full_train.tenure.max())


# get numerical value correlation with churn
# print(df_full_train[numerical].corrwith(df_full_train.churn))

"""
tenure           -0.344925 -> negative correlation
monthlycharges    0.188574 -> positive correlation
totalcharges     -0.193370 -> negative correlation
dtype: float64
"""


two_months_tenure_churn = df_full_train[df_full_train.tenure <= 2].churn.mean()
# print(two_months_tenure_churn)


two_months_more_churn = df_full_train[df_full_train.tenure > 2].churn.mean()
# print(two_months_more_churn)



