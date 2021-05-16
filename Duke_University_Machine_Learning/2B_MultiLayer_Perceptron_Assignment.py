#!/usr/bin/env python
# coding: utf-8

# # PyTorch Assignment: Multi-Layer Perceptron (MLP)

# **[Duke Community Standard](http://integrity.duke.edu/standard.html): By typing your name below, you are certifying that you have adhered to the Duke Community Standard in completing this assignment.**
# 
# Name: LIANG LI

# ### Multi-Layer Perceptrons
# 
# The simple logistic regression example we went over in the previous notebook is essentially a one-layer neural network, projecting straight from the input to the output predictions.
# While this can be effective for linearly separable data, occasionally a little more complexity is necessary.
# Neural networks with additional layers are typically able to learn more complex functions, leading to better performance.
# These additional layers (called "hidden" layers) transform the input into one or more intermediate representations before making a final prediction.
# 
# In the logistic regression example, the way we performed the transformation was with a fully-connected layer, which consisted of a linear transform (matrix multiply plus a bias).
# A neural network consisting of multiple successive fully-connected layers is commonly called a Multi-Layer Perceptron (MLP). 
# In the simple MLP below, a 4-d input is projected to a 5-d hidden representation, which is then projected to a single output that is used to make the final prediction.
# 
# <img src="Figures/MLP.png" width="300"/>
# 
# For the assignment, you will be building a MLP for MNIST.
# Mechanically, this is done very similary to our logistic regression example, but instead of going straight to a 10-d vector representing our output predictions, we might first transform to a 500-d vector with a "hidden" layer, then to the output of dimension 10.
# Before you do so, however, there's one more important thing to consider.
# 
# ### Nonlinearities
# 
# We typically include nonlinearities between layers of a neural network.
# There's a number of reasons to do so.
# For one, without anything nonlinear between them, successive linear transforms (fully connected layers) collapse into a single linear transform, which means the model isn't any more expressive than a single layer.
# On the other hand, intermediate nonlinearities prevent this collapse, allowing neural networks to approximate more complex functions.
# 
# There are a number of nonlinearities commonly used in neural networks, but one of the most popular is the [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)):
# 
# \begin{align}
# x = \max(0,x)
# \end{align}
# 
# There are a number of ways to implement this in PyTorch.
# We could do it with elementary PyTorch operations:

# In[1]:


import torch

x = torch.rand(5, 3)*2 - 1
x_relu_max = torch.max(torch.zeros_like(x),x)

print("x: {}".format(x))
print("x after ReLU with max: {}".format(x_relu_max))


# Of course, PyTorch also has the ReLU implemented, for example in `torch.nn.functional`:

# In[2]:


import torch.nn.functional as F

x_relu_F = F.relu(x)

print("x after ReLU with nn.functional: {}".format(x_relu_F))


# Same result.

# ### Assignment
# 
# Build a 2-layer MLP for MNIST digit classfication. Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:
# 
# Image (784 dimensions) ->  
# fully connected layer (500 hidden units) -> nonlinearity (ReLU) ->  
# fully connected (10 hidden units) -> softmax
# 
# Try building the model both with basic PyTorch operations, and then again with more object-oriented higher-level APIs. 
# You should get similar results!
# 
# 
# *Some hints*:
# - Even as we add additional layers, we still only require a single optimizer to learn the parameters.
# Just make sure to pass all parameters to it!
# - As you'll calculate in the Short Answer, this MLP model has many more parameters than the logisitic regression example, which makes it more challenging to learn.
# To get the best performance, you may want to play with the learning rate and increase the number of training epochs.
# - Be careful using `torch.nn.CrossEntropyLoss()`. 
# If you look at the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#crossentropyloss): you'll see that `torch.nn.CrossEntropyLoss()` combines the softmax operation with the cross-entropy.
# This means you need to pass in the logits (predictions pre-softmax) to this loss.
# Computing the softmax separately and feeding the result into `torch.nn.CrossEntropyLoss()` will significantly degrade your model's performance!

# In[3]:


### YOUR CODE HERE

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


from torch import nn
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))

model

print(model[0])
print(model.fc1)
# Make sure to print out your accuracy on the test set at the end.


# In[4]:


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


# In[5]:


transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


# In[6]:


from torchvision import datasets, transforms

mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True)


# In[7]:


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)


# In[8]:


plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');


# In[9]:


figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


# In[10]:


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)


# In[11]:


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss


# In[12]:


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


# In[13]:


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
       
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


# ### Short answer
# How many trainable parameters does your model have? 
# How does this compare to the logisitic regression example?

# In[16]:


images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))


# In[20]:


correct_count, all_count = 0, 0
for images,labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
        correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# `[Your answer here]`
