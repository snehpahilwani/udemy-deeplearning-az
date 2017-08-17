# Mega Case Study - Hybrid Learning Model

# SOM
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Feature scaling
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

# Visualizing the results
# Initialize the window
bone()
# Shows the neurons on the map , T gives the transpose
pcolor(som.distance_map().T)
# Legend
# Higher on the legend, more the probability being an outlier for the dataset
colorbar()
# Creating markers, red circle for outlier and green square for accepted
markers = ['o', 's']
colors = ['r','g']
# i is different indices in the dataset and x will be the vectors
for i, x in enumerate(X):
    # Winning node w
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None', markersize = 10, 
         markeredgewidth = 2) #the additional 0.5 puts the marker at the center of the square
show()    

# Finding the outliers
mappings = som.win_map(X) # dictionary of winning nodes
frauds = np.concatenate((mappings[(5,1)], mappings[(6,1)]), axis=0) # numbers keep changing everytime new SOM instance created
# zero for vertical concatenation
frauds = sc.inverse_transform(frauds)

# Going from unsupervised to supervised learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # all columns except the customer ID

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
        
        
# Feature scaling
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Creating ANN

# Initializing the ANN
classifier = Sequential()

# Adding the input layer & the first hidden layer with dropout
# (input + final output )/2 .. good recommendation for output dimensions
classifier.add(Dense(output_dim = 2, init = 'uniform', activation='relu', input_dim = 15))

# Adding final output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

# Compiling the ANN
# categorical_crossentropy loss in case of 3 or more categories of outputs
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=1)

# Predict fraud probabilities
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1],y_pred), axis=1) # one for vertical concatenation
y_pred = y_pred[y_pred[:,1].argsort()] #sorts numpy array by that column

