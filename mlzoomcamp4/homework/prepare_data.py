import pandas as pd




# preparation

df = pd.read_csv("CreditScoring.csv")
df.columns = df.columns.str.lower()

# de-code and encode some features


status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

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

# create the target variable
df['default'] = (df.status == "default").astype(int)

#drop the status
df.drop("status", axis=1, inplace=True)
