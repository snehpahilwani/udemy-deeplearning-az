# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data


# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Split into training and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) #this may change for other datasets
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting data into array with users in rows and movies in cols
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting ratings to binary ratings - Liked/Not Liked
# The limits seem weird though
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1
test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1

# Creating the neural network architecture
class RBM():
    """Contrastive divergence algorithm"""
    def __init__(self, nv, nh):
        """nv is number of visible nodes, nh is number of hidden nodes"""
        self.W = torch.randn(nh, nv) # Initializes tensor of size given in parantheses
        self.a = torch.randn(1, nh) # batch and bias parameters for the hidden nodes tensor
        self.b = torch.randn(1, nv) # batch and bias parameters for the visible nodes tensor
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) # Tensor multiplication for visible nodes and weight tensor
        activation = wx + self.a.expand_as(wx) # bias applied to each line of wx
        # Probability of hidden node given visible node
        p_h_given_v = torch.sigmoid(activation)
        #Bernoulli sampling(random threshold for binary selection) to return probability of hidden nodes given visible nodes
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W) # Matrix multiplication for hidden nodes and weight tensor
        activation = wy + self.b.expand_as(wy) # bias applied to each line of wy
        # Probability of visible node given hidden node
        p_v_given_h = torch.sigmoid(activation)
        #Bernoulli sampling(random threshold for binary selection) to return probability of visible nodes given hidden nodes
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk): 
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0-vk), 0) # trick to retain dimensions for b
        self.a += torch.sum((ph0-phk), 0)
       
nv = len(training_set[0])
nh = 100
batch_size = 100 # Online learning
rbm = RBM(nv,nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # Training loss
    s = 0. # Counter to normalize training loss
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size] #kth batch
        v0 = training_set[id_user:id_user + batch_size] #initial (not to change)
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] # To ensure training on existing ratings and not -1 ratings(NA)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0])) # Absolute difference for loss calculation
        s += 1.
    print('Epoch: ' + str(epoch) + ' Loss:' + str(train_loss/s))

# Testing the RBM
# MCMC - Markov Chain Monte Carlo Techniques

test_loss = 0 # Test loss
s = 0. # Counter to normalize test loss
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1] 
    vt = test_set[id_user:id_user + 1] 
    if len(vt[vt>=0]) > 0: # For existing ratings
        # Blind walk .. different than random walk as probabilities are random to start with
        # Only 1 round trip
        _,h = rbm.sample_h(v)   
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0])) # Absolute difference for loss calculation
        s += 1.
print('Test Loss: ' + str(test_loss/s))