import numpy as np

def split_the_data(X, y, train_split, val_split, test_split, seed):
    
    N = X.shape[0]
    idx = np.arange(N)
    
    np.random.seed(seed)
    np.random.shuffle(idx)

    
    # get length
    val_length = int((val_split)/100 * N)
    test_length = int((test_split)/100 * N)
    train_length = N - val_length - test_length

    # get index
    train_set = idx[:train_length]
    val_set = idx[train_length: train_length + val_length]
    test_set = idx[train_length + val_length:]

    # split and reset index
    X_train = X.iloc[train_set]
    y_train = y.iloc[train_set]

    X_val = X.iloc[val_set]
    y_val = y.iloc[val_set]

    X_test = X.iloc[test_set]
    y_test = y.iloc[test_set]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
