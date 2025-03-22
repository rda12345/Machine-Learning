#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network

The file contains my version of a neural network, coded by following the theory
in "Neural networks and deep learning by Michael Nielsen".

The neural network includes the following generalizations:
    1. Two possible initialization steps, the defult involves a normalized normal 
        distribution of weights and baises.
    2. It is possible to include a softmax normalization, within feedforward function
    3. A quadratic or cross-entropy cost function.
    4. Three possible options for regularization: NoRegularization, L1Regularization and L2Regularization.
        
"""
import numpy as np
import time
import random


class network(object):
    '''
    Initialization of a neural network
    
    lengths: list, containing the number of nodes in each level of the network
    
    '''
    def __init__(self,layers,CostFunction,Regularization):
        '''
        Initialization of the neural network
        
        Input
            layers: list, containing the number of nodes in each layer of the network
        
        Parameters
            num_layers: int, number of layers
            biases: list, including the biases of each layer (all of the layers except the first)
                            each one of the biases is a (layer[l],1) array
            weights: list, including the weights of each layer (all of the layers except the first)
                            each layer l has an associated array of dimension (layer[l],layer[l-1]).
        '''
        # Unrandomize for testing
        # np.random.seed(0)
        # random.seed(0)
        self.layers = layers 
        self.num_layers = len(layers)   # The layers are numbered from 1 to L = num_layers
        self.costfunction = CostFunction
        self.defult_initialization(layers)
        self.Regularization = Regularization
        
    def defult_initialization(self,layers):
        '''Intialization employing a normalized normal distribution.
            The normalization tends to allow the network to explore the parameter
            space and not get stuck at a part including a specific set of large parameters.
        '''
        self.biases =  [np.random.randn(y,1) for y in layers[1:]]
        self.weights =  [np.random.randn(x,y)/np.sqrt(x) for x,y in zip(layers[1:],layers[:-1])]
        
    def large_initialization(self,layers):
        '''Intialization employing a normal distribution. When the number of 
            nodes is large, there is a chance to randomly choose a set of parameters 
            with a relatively large magnitude. This may slow down the learning rate
            if the neural network gets stuck at a certain part of the parameter space.
        '''
        self.biases =  [np.random.randn(y,1) for y in layers[1:]]
        self.weights =  [np.random.randn(x,y) for x,y in zip(layers[1:],layers[:-1])]
        
        
    def feedforward(self,a,softmax = False):
        ''' Preforms the feedforward for sigmoid neorons.
            Retruns the output of the last layer, a^L.
            Softmax function can be added inorder to prevent a slow down in the 
            learning rate when the neorons saturate on the wrong prediction.
        '''
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            a = sigmoid(z)
        if softmax:
            a/np.linalg.norm(a)
        return a
        

    def SGD(self,training_data,epoches,mini_batch_size,eta,test_data = None,lam = 0):
        '''
        Stochastic Gradient Descent. The function applies the SGD algorithm
        with the backpropagation algorithm to update the weights of the
        neural network.
        
        Input
            training_data: list of tuples (x,y), where x is the example and y is the associated prediction
            mini_batch_size: int, the number of training examples in each mini batch
            eta: float, the step size
            regularization: function, the regularization function
            test_data: list of tuples (x,y), where x is the example and y is the associated prediction
        '''
        # Iteration over epoches
        for epoch in range(epoches):
            t1 = time.time()
            # Shuffling the training data to insure there is no structure
            random.shuffle(training_data)       # Modified to be the same as network.py this seed gives a better
                                                # starting value.
            # Iteration over the mini batches
            for q in range(0,len(training_data),mini_batch_size):
                mini_batch = training_data[q:q+mini_batch_size]
                self.update_weights(mini_batch,eta,lam,len(training_data))
            t2 = time.time()
            time_interval = t2 - t1
            if test_data != None:
                num_correct = self.evaluate(test_data)
                print('Epoch',epoch,'took',round(time_interval,2),'sec.',
                      str(num_correct),'/',str(len(test_data)))
    
    def update_weights(self,mini_batch,eta,lam,n):
        '''Updates the wieght using the backpropagation algorithm.'''
        m = len(mini_batch)
        # Set up the gradiant arrays
        dw =  [np.zeros(w.shape) for w in self.weights]     # A list of arrays
        db = [np.zeros(b.shape) for b in self.biases]       # A list of arrays
        # (x,y) = mini_batch[-1]
        # print(x.shape)
        # print(y.shape)
        for x,y in mini_batch:
            # For each example in a mini batch evaluate the change to the weights and biases
            Delta_dw, Delta_db = self.back_propagation(x,y)

            # For each weight (and bias) array sum contribution of the example to the change of the weight (bias)
            
            dw = [prev + change for prev,change in zip(dw,Delta_dw) ]
            db = [prev + change for prev,change in zip(db,Delta_db)]
            
            
        # Update the weights  #FIX HERE N AND LAM 
        self.weights = [w - (eta/m)*Delta_w
                        +(lam*eta/n)*self.Regularization.weight_correction(w) for w, Delta_w in zip(self.weights,dw)]
        self.biases = [b - (eta/m)*Delta_b for b, Delta_b in zip(self.biases,db)]

        
                
    def back_propagation(self,x,y):
        '''Feedforwards and then backpropagates, evaluating the correction to the 
            weights and baises.
        '''
        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # print('activation -1', activations[-1].shape)

        # Evaluating the last error: delta^{L}. Corresponds to Eq. (BP1) 
        delta = self.costfunction.delta(activations[-1],y,zs[-1])  
        deltas = [delta]
        
        # Backpropagate calculating the delta^{l}. Corresponds to Eq. (BP2)
        for k in range(self.num_layers-1,1,-1):
            delta = np.dot(self.weights[k-1].T,delta)*sigmoid_prime(zs[k-2])            
            deltas.append(delta)
        
        # Reverse the order of deltas
        deltas.reverse()
        
        # Evaluate the partial derivatives of the cost function w.r.t the biases and weights
        # Corresponds (BP3)
        Delta_db = deltas
        
        # Corresponds (BP4)
        Delta_dw = [np.outer(delt,act) for delt, act in zip(deltas,activations[:-1])]
        return (Delta_dw, Delta_db)
    
    def evaluate(self,test_data):
        '''Evaluates the number of correct predictions of the neural network.
           
            Input: 
                test_data: tuple, a pair (x,y), where x is the example and y is the correct classification.
            
            Returns:
                The number of correct predictions
        '''
        # For each figure in the data set check if it is equal pridiction, then sum over the number
        # of True results.
        num_correct = sum([np.argmax(self.feedforward(x))==y for (x,y) in test_data])
        return num_correct
        
        
            



class QuadCostFunction(object):
    
    @staticmethod
    def value(a,y):
        '''Returns the value of the quadratic cost function'''
        return np.linalg.norm(a-y)**2/2
    
    @staticmethod
    def delta(a,y,z):
        '''Returns the value of delta^L (Eq. (BP1)), associated with the quadratic cost function'''
        return (a-y)*sigmoid_prime(z)
    

class CrossEntropyCostFunction(object):
    
    @staticmethod
    def value(a,y):
        '''Returns the value of the cross-entropy cost function.
            The function uses nan_to_num method of numpy to fix the numerical
            instablity in the case where a approx 1.0.
        '''
        return np.sum(np.nan_to_num(-(y*np.log(a)+(1-y)*np.log(1-a))))
    
    @staticmethod    
    def delta(a,y,z):
        '''Returns the value of the quadratic cost function'''
        return a-y
        
        
class L1Regularization(object):
    
    @staticmethod 
    def weight_correction(w):
        '''Adds the correction due to the L1 regularization to \partial C/\partial w '''
        return -np.sign(w)
                
class L2Regularization(object):
    
    @staticmethod 
    def weight_correction(w):           
        '''Adds the correction due to the L1 regularization to \partial C/\partial w '''
        return -w
                
class NoRegularization(object):
   
    @staticmethod 
    def weight_correction(w):
        return 0

# TASKS:
# Define the quadratic cost function
# Defined the cross entropy
# Introduce the softmax option
# Program the dropout method
# Introduce a regularization function for the wieghts


def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#def quad_regularization(w,lam):
    

    
        
#Check 
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
layers = [784,30,10]
costfunction = QuadCostFunction()
#costfunction = CrossEntropyCostFunction()
net = network(layers,costfunction,L2Regularization())
epoches = 10
mini_batch_size = 10
eta = 3.0
lam = 1
net.SGD(training_data, epoches, mini_batch_size, eta,test_data,lam)


