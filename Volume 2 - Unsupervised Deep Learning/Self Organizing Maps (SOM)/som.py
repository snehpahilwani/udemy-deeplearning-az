# SOM

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
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
frauds = np.concatenate((mappings[(2,3)], mappings[(8,3)]), axis=0)
frauds = sc.inverse_transform(frauds)
