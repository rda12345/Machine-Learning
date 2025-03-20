#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradiant descent algorithms
"""

def grad_descent(theta_init,eta,J,gradJ,eps):
    ''' Gradiant descent algorithm. Providing the solution within accuracy eps.
        
        Input
            theta_int: array, including the initial guess for theta
            eta: float, step size
            J: objective function
            eps: float, error
        
        Returns
            theta: array, the converged classifer
    '''
    indicator = True
    t = 0
    theta = theta_init
    theta_new = None
    while theta_new == None or abs(J(theta_new) - J(theta)) >= eps:
         t += 1
         theta_new = theta - eta*grad(J,theta)
    return theta_new,t
    


def LR_grad_descent(theta_init,theta0_init,eta,J,gradJ,gradJ0,eps):
    ''' Logistic regression gradiant descent algorithm.
        Providing the solution within accuracy eps.
        
        Input
            theta_int: array, including the initial guess for theta
            eta: float, step size
            J: objective function
            eps: float, error
        
        Returns
            theta: array, the converged classifer
    '''
    indicator = True
    t = 0
    theta = theta_init
    theta0 = theta0_init
    theta_new = None
    while theta_new == None or abs(J(theta_new,theta0_new) - J(theta,theta_0)) >= eps:
         t += 1
         theta_new = theta - eta*gradJ(theta,theta0)
         theta0_new = theta - gradJ0(theta,theta0)
    return theta_new,t
    

def negative_log_likelyhood(theta,theta0):
    

             
            
