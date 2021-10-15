from prepare_data import (
        df_train, df_test, df_val)

from sklearn.feature_extraction import DictVectorizer

y_train = df_train.price.values
y_test = df_test.price.values
y_val = df_val.price.values

df_train = df_train.drop("price", axis=1)
df_test = df_test.drop("price", axis=1)
df_val = df_val.drop("price", axis=1)


def vectorizer(train, test, val):

    train = train.to_dict("records")
    test = test.to_dict("records")
    val = val.to_dict("records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train)

    X_test = dv.transform(test)

    X_val = dv.transform(val)

    return X_train, X_test, X_val, dv


X_train, X_test, X_val, dv = vectorizer(df_train, df_test, df_val)






