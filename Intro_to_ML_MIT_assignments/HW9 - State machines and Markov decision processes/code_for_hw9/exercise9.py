#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 9: State machines and Markov decision processes
"""
import numpy as np

## Q1 State machines

s_0 = 0
x =  [0, 1, 2, 1]
def f(s, x_i): 
    return max(s, x_i)

def g(s):
    return s * 2

def state_machine(x,f,g,s0):
    s = s0
    y_list = []
    for x_i in x:
        st = f(s,x_i)
        yt = g(st)
        s = st
        y_list.append(yt)
    return y_list


print(f'Q1A: {state_machine(x, f, g, s_0)}')


s_0 = (0, 0)
def f2(s, x_i): 
    return (s[0] + x_i, s[1] + 1)
def g2(s):
    return s[0] / s[1]
x = [0, 1, 2, 1]
  

print(f'Q1B: {state_machine(x, f2, g2, s_0)}')


## Q2 Markov decision processes

# The transition matrices of actions 'b' and 'c'
Tb = np.array([[0.0, 0.9, 0.1, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])
Tc = np.array([[0.0, 0.1, 0.9, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])

R = np.array([[0,1,0,2]]).T

V2 = R + np.sum(Tc@R,axis = 1, keepdims = True)

print(f'Q2B: {V2.T.tolist()[0]}')
    
    
      
        