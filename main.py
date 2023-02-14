# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Dropout

aapl_dataset = pd.read_csv('./dataset/AAPL.csv')

def create_model():
    model = keras.Sequential()

    # Add a hidden layer with 64
    model.add(keras.Input(shape=(16,)))

    # Add a dense layer with 64 units and ReLU activation function
    model.add(Dense(64, activation='relu', input_dim=3))

    # Add a hidden layer
    model.add(Dense(64, activation='relu'))

    # Final layer
    model.add(Dense(1, activation='linear'))

    # Compile the model with mean squared error loss and the Adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train_model(model, data, epochs, batch_size):m
    # history = model.fit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    print(aapl_dataset.head(10))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
