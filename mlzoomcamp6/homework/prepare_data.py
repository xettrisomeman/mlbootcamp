import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



columns = [
    'neighbourhood_group', 'room_type', 'latitude', 'longitude',
    'minimum_nights', 'number_of_reviews','reviews_per_month',
    'calculated_host_listings_count', 'availability_365',
    'price'
]

df = pd.read_csv('AB_NYC_2019.csv', usecols=columns)
df.reviews_per_month = df.reviews_per_month.fillna(0)


# apply the log transform to price


def price_log(price):
    return np.log1p(price)

# print(price_log(0))
df.price = df.price.map(price_log)
# print(df.price)

df_full_train, df_test = train_test_split(df, test_size = 0.20, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)









