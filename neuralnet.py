class1 = 3
class2 = 10

# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # raise NotImplementedError("You need to write this part!")

        self.in_size = in_size
        self.out_size = out_size

        '''
        # Initialize Neural Nets Model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_size)
        )
        '''

        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.MaxPool2d(5, 5),
            nn.Conv2d(32, 16, 5),
            nn.Flatten(),
            nn.Linear(16, 2),
            #nn.Linear(2, 2),
            #nn.Linear(2, out_size),
        )

        # Initialize Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lrate)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

        # Reshape the input
        #torch.reshape(x, (-1, 3, 32, 32))

        # Declare a Tensor output
        y_pred = torch.zeros([x.shape[0], self.out_size])

        # Determine predicted values y based on a given input x
        for i in range(0, x.shape[0]):
            y_pred[i] = self.model(x[i].view(-1, 3, 32, 32))
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        '''

        return y_pred

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        # raise NotImplementedError("You need to write this part!")

        # Zero all gradients for variables that will be updated
        self.optimizer.zero_grad()

        # Forward Pass: Calculates predicted values of y given input x
        y_pred = self.forward(x)

        # Calculate the loss (specialized function that determines the difference between y_pred and y)
        loss = self.loss_fn(y_pred, y)

        # Backward Pass: Calculates gradients of the loss with respect to "model parameters"?
        loss.backward()

        # Updating parameters (param -= learning_rate * param_grad)
        self.optimizer.step()

        # Return loss
        return loss.item()

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    # Initialize a list to store losses
    losses = []

    # Initialize a list to store predicted y labels for the dev set
    yhats = []

    # Normalize training set and development set by subtracting the mean and dividing by the std
    train_set = (train_set - torch.mean(train_set, 0)) / torch.std(train_set, 0)
    dev_set = (dev_set - torch.mean(dev_set, 0)) / torch.std(dev_set, 0)

    '''Training the Model'''
    # Initialize a NeuralNet object with the following parameters to train the model
    # Parameters:
    # - Learning Rate: 0.01
    # - Loss Function: Calculated by CrossEntropyLoss
    # - in_size: shape of train.set --> (N, in_size): choose shape[1]
    # - outsize: 2

    net = NeuralNet(0.01, nn.CrossEntropyLoss(), train_set.shape[1], 2)

    # Determine losses per iteration
    iteration = 0
    while iteration != n_iter:
        for i in range(0, train_set.shape[0], batch_size):
            losses.append(net.step(train_set[i: i + batch_size], train_labels[i: i + batch_size]))

            # Increment iterations
            iteration += 1

            # Exit if iterations == n_iter
            if iteration == n_iter:
                break

    '''Testing the Model'''
    # Disable Gradient Storing
    with torch.no_grad():
        for labels in dev_set:
            # Find the greater probability: P(image = animal) vs P(image = not animal)
            if torch.argmax(net(labels.view(-1, labels.shape[0]))) == 1:
                # Store 1 --> animal
                yhats.append(1)
            else:
                # Store 0 --> not animal
                yhats.append(0)

    return losses, yhats, net
