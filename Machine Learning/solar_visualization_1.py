import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

# Load CSV and columns
# List of all the columns to be compared with the Solar Intensity
label_list = ['Hour', 'Cloud Type', 'Dew Point', 'Solar Zenith Angle', 'Wind Speed', 'Precipitable Water', 'Relative Humidity', 'Temperature']

# Reading the csv file
df = pd.read_csv("83556_34.05_-118.26_2014.csv")

# fig, graph1 = plt.subplots(2,4)

i = 0
j = 0
# for loop to plot Intensity vs different factors to find relationship between them
for label in label_list:
    X = np.array(df[label])     # Different factors are read into X
    Y = np.array(df['GHI'])     # GHI read into Y

    # Make X and Y a 1 Dimensional Numpy Array
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)

    # Split the data into training/testing sets
    X_train = X[:-250]
    X_test = X[-250:]

    # Split the targets into training/testing sets
    Y_train = Y[:-250]
    Y_test = Y[-250:]

    # Plot outputs
    plt.scatter(X_test, Y_test, color='black')      # Scatter Plot using PyPlot
    plt.title('Test Data')
    plt.xlabel(label)
    plt.ylabel('GHI')
    plt.xticks(())
    plt.yticks(())

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # graph1[i % 2, j % 4].plot(X_test, regr.predict(X_test), color='red', linewidth=3)
    plt.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
    i = i + 1
    j = j + 1
    #plt.axis([min(X_test), max(X_test), min(Y_test), max(Y_test)])

    plt.show()