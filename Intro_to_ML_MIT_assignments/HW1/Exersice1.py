#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 1
"""
import numpy as np


def length(col_v):
    ''' returns the norm of a column vector
        
        Input
        col_v: array: (d,1) dimension
        
        Returns
        norm: float
    '''
    return np.sqrt(col_v.T@col_v)[0][0]

# a = np.array([[1],[1]])
# print(length(a))

def signed_dist(x, th, th0):
    ''' Evaluates the signed disatance from a hyperplane defined by th and th0 to a point x.
        The distance is the projection from the point x to the plane, it is positive above the
        hyperplane as defined w.r.t the direction of th, and negative bellow the plan.
        
        Input
        x: (d,1) array
        th: (d,1) array
        th0: float
        
        Returns
        An (1,1) array containing the signed distance
        '''
    return    ((th.T@x+th0)/length(th))


def positive(x, th, th0):
    ''' Evaluates the side of the point w.r.t the hyperplane.
        1: same side, 0 on the hyperplane; -1 on the oposite side.
    '''
    return np.sign(signed_dist(x, th, th0))


def score(data, labels, th, th0): 
    return np.sum((labels == positive(data, th, th0).astype(int))[0])


def score_mat(data, labels, ths, th0s): 
    return np.sum(labels == positive(data, ths, th0s.T),axis = 1,keepdims = True)

def best_separator(data, labels, ths, th0s):
    '''
    Input
    data: (d,m) array, representing n data points of d dimensions.
    labels: (d,1) array of elements in (+1, -1), representing target labels
    ths: a d by m array of floats, representing m candidate th's 
    th0s: (1,m) array of floats, corresponding to the m canidate th0
        
    Returns
    Tuple with the pair (th,th0) with the best score.
    '''
    scores = score_mat(data, labels, ths, th0s)
    ind = np.argmax(scores)
    th_best = ths[:,ind]
    th0_best = th0s[:,ind]
    return (th_best,th0_best)
    
## Test
data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
labels = np.array([[-1, -1, +1, +1, +1]])
ths = np.array([[1,1],[1,2]]).T
th0s = np.array([[-2,2]])
print(best_separator(data, labels, ths, th0s))








## Numpy exercise
# a = np.array([[1,1]])
# #print(np.shape(a.T))
# mat = [[1,2,3],[4,5,6],[7,8,9]]
# mat_array = np.array(mat)
# #print(np.sum(mat,axis = 0)) # sum of the columns
# #print(np.sum(mat,axis = 1, keepdims= True)) # sum of rows while keeping dimensions

# C = np.array([[1,2,3,4]]).T
# a = C*C
# #print(C.shape)
# #print(np.dot(C.T,C).shape)
# #print(C[:,0].shape)

# A = np.array([[1,2,3],[2,3,4],[3,4,5],[1,2,3]])
# #print(A[:,1:2])
# col_v = np.array([[1,1,1]]).T
# #print(np.linalg.norm(col_v))
# print(A[:,-1:])
# #print(np.sqrt(np.dot(col_v.T,col_v))[0][0])
# data = np.array([[150,5.8],[130,5.5],[120,5.3]])
# b = np.dot(data,np.array([[1, 1]]).T)
# print(b)