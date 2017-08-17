# RNN

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing RNN libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# Importing libraries for evaluation
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values
#Importing the test set
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Feature scaling
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting inputs and outputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))

# Building the RNN
# Initializing the RNN
regressor = Sequential()

# Adding input layer and LSTM layer
# None to accept any timesteps, just one feature which is the stock price hence, the second param is 1 in input_shape
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding output layer (stock price at time t+1)
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# Predictions 
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizations
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.xlabel('Time step')
plt.ylabel('Stock price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show()

# Evaluating the RNN
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
