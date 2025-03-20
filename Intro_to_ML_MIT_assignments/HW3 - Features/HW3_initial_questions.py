#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW3_initial_part.py
"""
import numpy as np
import math

def perceptron_origin(data, labels, params={}, hook=None):
    ''' The preceptron algorithm evaluates the linear seperator (if it exists)
        of a a data set.
        
        Input
            data: d by n array, containing n feature vectors
            labels: 1 by n array, the corresponding labels
            params: dict, specifying extra parameters to this algorithm
            hook: either None or a function that takes the tuple (th, th0)
                as an argument and displays the separator graphically.
                
        Parameters
            T: int, the maximum number of interations until termination.
        
        Returns
        Tuple (theta,theta_0)
        th: d by 1 array, with the seperator
        th0: 1 by 1 array containing the offset
    '''
    # if T not in params, default to 100
    T = params.get('T', 300000)
    d, n = data.shape
    th = np.zeros((d,1))
    counter = 0
    for t in range(T):
        check = True
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if np.sign(y*(th.T@x)) <= 0.0:
                th += y*x
                counter += 1
                check = False
        # If all conditions are met the algorithm has converged, so abort.
        if check:    
            print('Algorithm converged')
            break
    return th,counter




def perceptron(data, labels, params={}, hook=None):
    ''' The preceptron algorithm evaluates the linear seperator (if it exists)
        of a a data set.
        
        Input
            data: d by n array, containing n feature vectors
            labels: 1 by n array, the corresponding labels
            params: dict, specifying extra parameters to this algorithm
            hook: either None or a function that takes the tuple (th, th0)
                as an argument and displays the separator graphically.
                
        Parameters
            T: int, the maximum number of interations until termination.
        
        Returns
        Tuple (theta,theta_0)
        th: d by 1 array, with the seperator
        th0: 1 by 1 array containing the offset
    '''
    # if T not in params, default to 100
    T = params.get('T', 300000)
    d, n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    counter = 0
    for t in range(T):
        check = True
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if np.sign(y*(th.T@x+th0)) <= 0.0:
                th += y*x
                th0 += y
                counter += 1
                check = False
        # If all conditions are met the algorithm has converged, so abort.
        if check:    
            print('Algorithm converged')
            break
    return th, th0


#data = np.array([[200, 800, 200, 800],
             #[0.2,  0.2,  0.8,  0.8],[1,1,1,1]])
# data = np.array([[0.2, 0.8, 0.2, 0.8],
#              [0.2,  0.2,  0.8,  0.8],[1,1,1,1]])
# labels = np.array([[-1, -1, 1, 1]])

# data =   np.array([[2, 3,  4,  5]]) 
# labels = np.array([[1, 1, -1, -1,]])

data =   np.array([[1,2, 3,  4,  5, 6]]) 
labels = np.array([[1, 1, -1, -1, 1, 1]])
# th, th0 = perceptron(data,labels,params={}, hook=None)
# print((th,th0))


def one_hot(x, k):
    vec = np.zeros((k,1))
    vec[x-1,0] = 1
    return vec

l = []
k = data.shape[1]
for i in range(1,k+1):
    l.append(one_hot(i, k))
    
data_one_hot = np.concatenate(tuple(l),axis = 1)
print(data_one_hot)
# print(data_one_hot)
th, th0 = perceptron(data_one_hot,labels,params={}, hook=None)
print((th,th0))


from HW3 import make_polynomial_feature_fun

make_polynomial_feature_fun(1)


# th = np.array([[0,2,1,-2,-1,0]]).T
# length_th = math.sqrt(th.T@th)
# sam = np.array([[1,0,0,0,0,0]]).T
# nok = np.array([[0,0,0,0,0,1]]).T
#print(th.T@sam)
#print(th.T@nok)

# sam_dist = th.T@sam/length_th
# nok_dist = th.T@nok/length_th
# print('sam dist: ',sam_dist)
# print('nok dist: ',nok_dist)
#Check
# d, n = data_one_hot.shape
# for i in range(n):
#     x = data_one_hot[:,i:i+1]
#     y = labels[:,i:i+1]
#     print(np.sign(y*(th.T@x+th0)))
# #Q1
# th = np.array([[0,1,-0.5]]).T
# gammas = []
# Rs = []
# length_th = math.sqrt((th.T@th)[0,0])
# for i in range(3):
#     elem = th.T@data[:,i]*labels[:,i]/length_th
#     gammas.append(elem)
    
    
# print(gammas)
# print('gamma: ' ,min(gammas))
# gamma = min(gammas)[0]    
# R = math.sqrt(0.8**2+0.8**2+1)

# error_bound = (R/gamma)**2  
# print('error bound: ',error_bound)

# th, num_errors = perceptron_origin(data, labels, params={}, hook=None)
# print(num_errors)
# print(th)

# data = np.array([[200, 800, 200, 800],
#              [0.2,  0.2,  0.8,  0.8],[1000,1000,1000,1000]])*0.001
# print(data)
# th = np.array([[0,1,-0.0005]]).T
# gammas = []
# Rs = []
# length_th = math.sqrt((th.T@th)[0,0])
# for i in range(3):
#     elem = th.T@data[:,i:i+1]*labels[:,i:i+1]/length_th
#     gammas.append(elem)
    
# gamma = min(gammas)[0,0] 
# print(gammas)
# print(gamma)
# R = math.sqrt(0.8**2+0.0008**2+1)
# error_bound = (R/gamma)**2  
# print(error_bound)


