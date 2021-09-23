from eda import df_full_train



# churn rate
global_churn = df_full_train.churn.mean()
# print(global_churn)


# churn rate with each groups

# funtion to get churn

def get_churn_ratio(columns, data):
    churn_rate = df_full_train[df_full_train[columns] == data].churn.value_counts(normalize=True).iloc[1]
    return churn_rate

# gender
#print(df_full_train.gender.value_counts())

churn_female = get_churn_ratio("gender", "female")
# print(churn_female)
churn_male = get_churn_ratio("gender", "male")
# print(churn_male)


# partner 
# print(df_full_train.partner.value_counts())

churn_partner_no = get_churn_ratio("partner", "no")
# print(churn_partner_no)
churn_partner_yes = get_churn_ratio("partner", "yes")
# print(churn_partner_yes)


# return churn dataframe

# gender_churn = df_full_train.groupby("gender").churn.mean()
# df_group = df_full_train.groupby("gender").churn.agg(["mean", "count"]) # add different values

# print(df_group)



