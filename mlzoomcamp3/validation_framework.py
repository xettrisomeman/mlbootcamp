from data_preparation import df

from sklearn.model_selection import train_test_split

# split the data to train and test
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=42)



# now split train dataset to (train and validation)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25 ,random_state=42)



# print the shape of the splitted dataset
# print(df_train.shape, df_val.shape, df_test.shape)


# reset the index of the dataset, make it sequential
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)


# now get y variable
y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values


# drop churn

df_train = df_train.drop("churn", axis=1)
df_val = df_val.drop("churn", axis=1)
df_test = df_test.drop("churn", axis=1)



# print(df_train.shape, y_train.shape)
# print(df_test.shape, y_test.shape)
# print(df_val.shape, y_val.shape)
