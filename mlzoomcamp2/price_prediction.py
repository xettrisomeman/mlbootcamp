import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# read the data
df = pd.read_csv("data.csv")
#print(df.shape)
#print(df.head())

#make lowercase
df.columns = df.columns.str.lower()
# print(df.head())

# change space with underscore
df.columns = df.columns.str.replace(" ", "_")
# print(df.head())

# normalize the values 
category_columns = list(df.dtypes[df.dtypes == "object"].index)
df[category_columns] = df[category_columns].apply(lambda x: x.str.lower().str.replace(" ", "_"))
# print(df.head())



# Exploratory Data Analysis

"""
for col in df.columns:
    # print(col)
    # print(df[col].head())
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()
"""

# Distribution of price

# sns.histplot(df['msrp'].values, bins=50)
# plt.show()

"""
price_logs = df.msrp.map(np.log1p)
print(price_logs)

sns.histplot(price_logs, bins=50)
plt.show()
"""

# check missing values
# print(df.isna().sum())



# SET UP VALIDATION FRAMEWORK

n = len(df)

n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test
# print(n_train, n_val, n_test)

# generate list of indexes
idx = np.arange(n)

# shuffle the index 
np.random.seed(2)
np.random.shuffle(idx)
# print(idx)

# now split the datset
df_train = df.iloc[idx[:n_train]]
df_test = df.iloc[idx[n_train: n_train + n_test]]
df_val = df.iloc[idx[n_train + n_val:]]
# print(df_train.shape, df_val.shape, df_test.shape)


# reset the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# log transform the y variable: log(y+1)
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
# print(y_train, y_test, y_val)


# drop the msrp values
df_train.drop(columns=["msrp"], inplace=True)
df_val.drop(columns=["msrp"], inplace=True)
df_test.drop(columns=["msrp"], inplace=True)


