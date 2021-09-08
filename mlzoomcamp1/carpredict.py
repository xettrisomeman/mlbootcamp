import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')


# print("The head of the data")
# print(df.head())

def get_car_average_price(dataframe, car_name):

    # get bmw cars "msrp" values
    bmw_cars = dataframe[dataframe['Make'] == car_name]['MSRP'].values

    # iterate over bmw prices
    bmw_cars_price = [car for car in bmw_cars]

    # return average price
    return sum(bmw_cars_price)/bmw_cars.shape[0]

print(f"Question 3.")
print(get_car_average_price(df, "BMW"))


def get_missing_values(dataframe, date):

    # get missing data before and after 2015
    after_2015 = dataframe[dataframe['Year'] >= date]

    # check null values
    get_empty_column = after_2015.isna().sum().loc['Engine HP']

    # return null value after 2015 and equal to 2015
    return get_empty_column

print(f"Question 4.")
print(get_missing_values(df, 2015))


def get_average_and_fill(dataframe):

    # check mean value of engine_HP
    engine_HP_mean = dataframe.describe().loc['mean', 'Engine HP']

    #fill the missing value with mean value
    fill_engine_HP = dataframe.fillna(engine_HP_mean, axis=1)

    # get mean value after filling the data
    get_mean_again = dataframe.describe().loc['mean', 'Engine HP']

    # return before mean and after filling the null values
    return (f"Before mean : {round(engine_HP_mean)}", f"After mean: {round(get_mean_again)}")

print(f"Question 5.")
print(get_average_and_fill(df))


def get_roll_royce(dataframe, car_name, columns):

    # get roll royce
    get_royce = dataframe[dataframe['Make'] == car_name]

    # get the required columns
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
print("Question 6.")
print(sum_InvX)




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

print("Question 7.")
print(get_normal_equation(x, inv_xTx, y))
