# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2

aapl_dataset = pd.read_csv('./dataset/AAPL.csv')
aapl_dataset = shuffle(aapl_dataset)
aapl_dataset.reset_index(inplace=True, drop=True)
aapl_dataset[["day", "month", "year"]] = aapl_dataset['Date'].str.split('-', expand=True).astype(float)

# split into featuers and labels
features = aapl_dataset[["day", "month", "year"]]

# get the mean, std, and z-score of features and label
features_mean = features.mean()
features_std = features.std()
features_Z = (features - features_mean) / features_std

label = aapl_dataset["Low"]

label_mean = label.mean()
label_std = label.std()
label_Z = (label - label_mean) / label_std


# split up into training and test data
x_train, x_test, y_train, y_test = train_test_split(features_Z, label_Z, test_size=0.2)


def create_model(learning_rate):
    model = keras.Sequential()

    # Add a dense layer with 64 units and ReLU activation function
    model.add(Dense(64, activation='relu', input_dim=3, kernel_regularizer=l2(0.01)))

    # Add a hidden layer
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))

    # Final layer
    model.add(Dense(1, activation='linear'))

    # Compile the model with mean squared error loss and the Adam optimizer
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

    return model


# HYPER PARAMETERS
batch_size = 400
epochs = 100
learning_rate = 0.3

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print("test")
    # print(len(label))
    # print(aapl_dataset.dtypes)

    model = create_model(learning_rate)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
    model.evaluate(x_test, y_test, batch_size=batch_size)

    prediction_df = pd.DataFrame(data=[[14, 2, 2025]], columns=['day', 'month', 'year'])

    #normalize value to training data
    prediction_df_norm = (prediction_df - features_mean) / features_std

    # z = (x - u)/std
    # to calculate prediction output, we need to reverse the z score and get the actual value
    # stock_predicted_value = model.predict(prediction_df_norm) * features_std + features_mean
    stock_predicted_value_Z = model.predict(prediction_df_norm)[0][0]
    stock_predicted_value = stock_predicted_value_Z * label_std + label_mean
    print("prediction: " + str(stock_predicted_value))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
