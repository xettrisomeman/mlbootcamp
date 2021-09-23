import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# check data
df = pd.read_csv("Churn.csv")

# check columns by transposing the data.
# print(df.head().T)

# replace (columns name) space with underscore
df.columns = df.columns.str.lower().str.replace(" ", "_")

# get all columns with categorical values
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

# print(categorical_columns)

# loop over the categorical value columns
for columns in categorical_columns:
    # switch space with underscore
    df[columns] = df[columns].apply(lambda x: x.replace(" ", "_"))
    # lower the string
    df[columns] = df[columns].apply(lambda x: x.lower())


#check datatypes of pandas columns
# print(df.dtypes)

#change object datas to int(columns totalcharges is shown as object whereas the value it contains is integer)

# errors -> coerce , it means leave the value to null
total_charges = pd.to_numeric(df.totalcharges, errors="coerce")

# fill null values with 0
df.totalcharges = total_charges.fillna(0)

# check if the null value still exist
# print(df.totalcharges.isnull().sum())


# Check churn

# print(df.churn.value_counts())

# change yes/no to (1, 0)
df.churn = df.churn.map({"yes": 1, "no": 0})

# print value counts after the change
# print(df.churn.value_counts())

# print(df.isnull().sum())
