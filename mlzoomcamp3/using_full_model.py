from logscikit import logistic_regression
from validation_framework import df_full_train, df_test, y_test
from eda import categorical, numerical


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


dicts_full_train = df_full_train[categorical + numerical].to_dict("records")


dv = DictVectorizer(sparse=False)


X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = df_full_train.churn
# print(y)

model = LogisticRegression(max_iter=1000)

model.fit(X_full_train, y_full_train)

dicts_test = df_test[categorical + numerical].to_dict("records")

X_test = dv.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:, 1]

churn_decision = y_pred >= 0.5


# print((churn_decision == y_test).mean())


# Take single item
item_number = 100
customer = dicts_test[item_number]


# transform
customer_dic = dv.transform(customer)
# print(customer_dic.shape)


# predict
y_pred = model.predict(customer_dic)
print(y_pred[0] == y_test[item_number])



