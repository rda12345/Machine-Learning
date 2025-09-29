#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preceptron algorithm
The algorithm finds the hypothesis function for a set of data points which are
linearly separable (which can be seprated by a linear line.
                    
The precptron_origin function assumes that the hypothesis goes through the origin (theta_0 = 0).
"""
import numpy as np
def preceptron_origin(D,T):
    ''' Returns a linear hypothesis function trough the origin for the data set D.
        
        Input
            D: tuple, containing (X,y)
            X: 2D array, columns of the array contain the features of each sample
            y: 2D array, row vector with the corresponding labels
            T: int, number of iterations until convergence or completion
            
        Returns
            theta: 2D array, column vector with the best or converged hypothesis.
    '''
    (X,y) = D
    (d,n) = np.shape(X)
    theta = np.zeros((d,1))
    for t in range(T):
        check = True
        for i in range(n):
            if y[0,i]*(theta.T@X[:,i]) <= 0:
                theta += y[0,i]*X[:,i:i+1]
                check = False
        if check:
            break
    return theta



def preceptron(D,T):
    ''' Returns a linear hypothesis function trough the origin for the data set D.
        
        Input
            D: tuple, containing (X,y)
            X: 2D array, columns of the array contain the features of each sample
            y: 2D array, row vector with the corresponding labels
            T: int, number of iterations until convergence or completion
            
        Returns
            theta: 2D array, column vector with the best or converged hypothesis.
    '''
    (X,y) = D
    (d,n) = np.shape(X)
    theta = np.zeros((d,1))
    theta_0 = 0 
    for t in range(T):
        check = True
        for i in range(n):
            if y[0,i]*(theta.T@X[:,i]) <= 0:
                theta += y[0,i]*X[:,i:i+1]
                theta_0 += y[0,i]
                check = False
        if check:
            break
    return theta, theta_0
                
## Runs
# X = np.array([[1,-1],[0,1],[-1.5,-1]]).T
# #X = np.array([[0,1],[-10,-1],[1,-1]]).T
# y = np.array([[1,-1,1]])
T = 20
# D = (X,y)
# #print(np.shape(X[:,0]))
# theta = preceptron_origin(D, T)

## Q3
# X = np.array([[-3,2],[-1,1],[-1,-1],[2,2],[1,-1]]).T
# y = np.array([[1,-1,-1,-1,-1]])
# D = (X,y)
# theta = preceptron_origin(D,T)
# print(theta)

## Q4
X = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]]).T
y = np.array([[-1,-1,-1,-1,-1,-1,-1,1]])
D = (X,y)
theta = preceptron(D,T)
print(theta)






