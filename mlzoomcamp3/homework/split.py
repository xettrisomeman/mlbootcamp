from sklearn.model_selection import train_test_split


def split_the_data(df_data):

    df_full_train, df_test = train_test_split(df_data, test_size = 0.2, random_state = 42)

    df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)


    # reset the columns
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)


    # get y values

    y_train = df_train.price
    y_test = df_test.price
    y_val = df_val.price


    # remove price columns

    df_train = df_train.drop("price", axis=1)
    df_test = df_test.drop("price", axis=1)
    df_val = df_val.drop("price", axis=1)

    return df_train, df_test, df_val, y_train, y_test, y_val






