from clean_text import clean_text_dataset

import joblib

from string import punctuation

import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# read the data
nepali_text = pd.read_csv("./nepali_news_classification.csv")
# print(nepali_text.head())


# remove unnecessary column
nepali_text.drop("headings", axis=1, inplace=True)

# note:- we are notchanging label to integer as it will be easier to get the prediction as "sports" than showing it as label 0


# clean the dataset
# read stop words, punctuation
# we will be using a fuction i have created in a different file (clean_text.py)

# functions to train and predict
def train(X_train, y_train, function):
    tfidf = TfidfVectorizer(tokenizer = function)

    # vectorize the train dataset
    x_train = tfidf.fit_transform(X_train)

    # create a model

    model = LogisticRegression(C = 10, solver="saga", multi_class="multinomial", max_iter=1000)

    model.fit(x_train, y_train)
    
    return model, tfidf


def predict(model, tfidf, X_val):

    # transform our validation dataset to vectors
    x_val = tfidf.transform(X_val)

    y_pred = model.predict(x_val)

    return y_pred


# now we split the dataset 60/20/20
# stratify, makes sure we get same propotion of data like we have in our csv file.
df_full_train, df_test = train_test_split(nepali_text, test_size = 0.2, random_state = 1, stratify = nepali_text.label)

df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1, stratify = df_full_train.label)


# check if the label is equally distributed or not
# print(df_train.label.value_counts(), df_val.label.value_counts())


# create x_train, y_train and x_val, y_val
X_train = df_train.paras.values
y_train = df_train.label

X_val = df_val.paras.values
y_val  = df_val.label
# print(X_train, y_train)
# print(X_val, y_val)


# let's train our model
model, tfidf = train(X_train, y_train, clean_text_dataset)

# saving the model and tfidfvectorizer
joblib.dump(model, "model.bin")
joblib.dump(tfidf, "tfidf.bin")


# making a prediction
y_val_pred = predict(model, tfidf, X_val)


# print(f1_score(y_val, y_val_pred, average="macro"))











