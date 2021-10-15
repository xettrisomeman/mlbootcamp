
# Data Cleaning and Preparation
import pandas as pd


data = "./CreditScoring.csv"


df = pd.read_csv(data)

#make columns lower
df.columns = df.columns.str.lower()

# check values
# print(df.status.value_counts())

status_values = {1: "ok", 2: "default", 3: "unk"}
df.status = df.status.map(status_values)


home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

# print(df.head())


# prepare the numerical values
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=0)
    
# remove client with unknown default status
df = df[df.status != "unk"].reset_index(drop = True)


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 11)
df_train, df_val= train_test_split(df_full_train, test_size= 0.25 , random_state = 11)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = (df_train.status == "default").astype('int').values
y_val = (df_val.status == "default").astype('int').values
y_test = (df_test.status == "default").astype('int').values


del df_train['status']
del df_val['status']
del df_test['status']

# print(df_train.shape, y_train.shape)
# print(df_test.shape, y_test.shape)
# print(df_val.shape, y_val.shape)

