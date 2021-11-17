import tensorflow as tf
from tensorflow.keras import layers, models


def make_model():

    # making a sequential model
    model = models.Sequential()

    # model with 32 filer, kernal_size (3, 3,), activation= "relu"
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
              input_shape=(150, 150, 3)))

    # adding maxpooling layer with pool_size = (2, 2)
    model.add(layers.MaxPooling2D(2, 2))

    # flatting the output
    model.add(layers.Flatten())

    # dense layer with 64 neurons
    model.add(layers.Dense(64, activation="relu"))

    # last dense layer with 1 neuron (output)
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
