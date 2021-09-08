import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# print(df.head())

def get_car_average_price(dataframe, car_name):
    bmw_cars = dataframe[dataframe['Make'] == car_name]['MSRP'].values
    bmw_cars_price = [car for car in bmw_cars]
    return sum(bmw_cars_price)/bmw_cars.shape[0]

#Q3. print(get_car_average_price(df, "BMW"))

def get_missing_values(dataframe, date):
    after_2015 = dataframe[dataframe['Year'] >= date]
    get_empty_column = after_2015.isna().sum().loc['Engine HP']
    return get_empty_column

#Q4. print(get_missing_values(df, 2015))


def get_average_and_fill(dataframe):
    engine_HP_mean = dataframe.describe().loc['mean', 'Engine HP']
    fill_engine_HP = dataframe.fillna(engine_HP_mean, axis=1)

    get_mean_again = dataframe.describe().loc['mean', 'Engine HP']
    return (f"Before mean : {round(engine_HP_mean)}", f"After mean: {round(get_mean_again)}")

# Q5. print(get_average_and_fill(df))


def get_roll_royce(dataframe, car_name, columns):
    get_royce = dataframe[dataframe['Make'] == car_name]
    get_columns = get_royce[columns]

    # drop duplicates rows
    drop_duplicates_rows = get_columns.drop_duplicates()

    # get underlying numpy array
    x = drop_duplicates_rows.values

    # dot product
    xTx= np.matmul(x.T, x)

    # invert the dot product
    inv_xTx = np.linalg.inv(xTx)

    
    # sum of all the elements of the invert matrices
    sum_InvX = sum([sum(inv) for inv in inv_xTx])
    return inv_xTx, sum_InvX, x

inv_xTx, sum_InvX, x = get_roll_royce(df, "Rolls-Royce", ['Engine HP', 'Engine Cylinders', 'highway MPG'])
# print(sum_InvX)




y = np.array([
    1_000,
    11_00,
    9_00,
    12_00,
    1_000,
    850,
    13_00
])

def get_normal_equation(x, inv_xTx, y):
    matrix_mul = np.matmul(inv_xTx, x.T)

    # change the dimension of the arrray
    y = y.reshape(7, 1)
    
    # matmul under the hood
    # w -> [3, 7] * [7, 1] -> [3, 1]
    w2 = matrix_mul @ y
    
    return w2[0]


# Q7. print(get_normal_equation(x, inv_xTx, y))
