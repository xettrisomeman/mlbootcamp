from validation_framework import df_train, df_test, df_val
from eda import categorical, numerical

from sklearn.feature_extraction import DictVectorizer




# turn features into dictionary (first feature/columns have to be converted to dictionary to pass into dictvectorizer)
train_dicts = df_train[categorical + numerical].to_dict("records")
val_dicts = df_val[categorical + numerical].to_dict("records")
# print(train_dicts)
# print(val_dicts)
# print(test_dicts)


dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
# X_test = dv.transform(test_dicts)
# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)
