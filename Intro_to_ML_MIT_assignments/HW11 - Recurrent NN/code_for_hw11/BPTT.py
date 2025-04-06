#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPTT - Backpropagation through time.
used to train a recurrent neural network

Instructions:
Complete the following implementation of BPTT. This is a method in an RNN class.
The instance (self) has attributes that define the RNN, in particular:
The weight matrices and offsets: self.Wss, self.Wsx, self.Wo, self.Wss0 and self.Wo0.
The activation functions and their derivatives: self.f1, self.df1, self.f2, self.df2 
The loss function and the derivative of the combined loss and final activation 
(as we did for feedforward networks): self.loss_fn and self.dloss_f2. 
The dimensions of states (hidden), inputs and outputs: self.hidden_dim, self.input_dim, self.output_dim. 
The initial state and, the current state: self.init_state and self.hidden_state 
The step size for gradient descent: self.step_size
"""
import pdb
from util import *
import numpy as np
import sm
import functools
import random
import _pickle as cPickle





def bptt(self, xs, dLtdz2, states):
    dWsx = np.zeros_like(self.Wsx)
    dWss = np.zeros_like(self.Wss)
    dWo = np.zeros_like(self.Wo)
    dWss0 = np.zeros_like(self.Wss0)
    dWo0 = np.zeros_like(self.Wo0)
    # Derivative of future loss (from t+1 forward) wrt state at time t
    # initially 0;  will pass "back" through iterations
    dFtdst = np.zeros((self.hidden_dim, 1))
    k = xs.shape[1]
    # Technically we are considering time steps 1..k, but we need
    # to index into our xs and states with indices 0..k-1
    for t in range(k-1, -1, -1):
        # Get relevant quantities
        xt = xs[:, t:t+1]
        st = states[:, t:t+1]
        stm1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
        dLtdz2t = dLtdz2[:, t:t+1]
        # Compute gradients step by step
        # ==> Use self.df1(st) to get dfdz1;
        # ==> Use self.Wo, self.Wss, etc. for weight matrices
        # derivative of loss at time t wrt state at time t
        dLtdst = np.dot(self.Wo.T,dLtdz2t)        # Your code
        # derivatives of loss from t forward
        dFtm1dst = dLtdst + dFtdst 
        dFtm1dz1t = self.df1(st) * dFtm1dst
        dFtm1dstm1 = np.dot(self.Wss.T,dFtm1dz1t)
        # gradienst wrt weights
        dLtdWo = np.dot(dLtdz2t,st.T)
        dLtdWo0 = dLtdz2t
        dFtm1dWss = np.dot(dFtm1dz1t,stm1.T)
        dFtm1dWss0 = dFtm1dz1t 
        dFtm1dWsx = np.dot(dFtm1dz1t,xt.T)
        # Accumulate updates to weights
        dWsx += dFtm1dWsx
        dWss += dFtm1dWss
        dWss0 += dFtm1dWss0
        dWo += dLtdWo
        dWo0 += dLtdWo0
        # pass delta "back" to next iteration
        dFtdst = dFtm1dstm1
    return dWsx, dWss, dWo, dWss0, dWo0


def delay_num_test(bptt, delay = 1, num_epochs = 10000, num_seqs = 10000, seq_length = 10, step_size = .005):
  '''
  This is a test function provided to help you debug your implementation

  In case we want to initialize.  Now just for delay = 1
  # Wsx = np.array([[1.], [0.]])
  # Wss = np.array([[0., 0.],
  #              [1., 0.]])
  # Wo = np.array([[0., 1.]])
  # Wss0 = np.array([[0.], [0.0]])
  # Wo0 = np.array([[0.]])
  '''
  np.random.seed(0)
  data = []
  for _ in range(num_seqs):
    vals = np.random.random((1, seq_length))
    x = np.hstack([vals, np.zeros([1, delay])])
    y = np.hstack([np.zeros((1, delay)), vals])
    data.append((x, y))
  m = (delay + 1) * 2
  RNN.bptt = bptt
  rnn = RNN(1, m, 1, quadratic_loss, lambda z: z, quadratic_linear_gradient, step_size, lambda z: z, lambda z: 1)
  # Wsx = Wsx, Wo = Wo, Wss = Wss, Wo0 = Wo0, Wss0 = Wss0)
  rnn.train_seq_to_seq(data, num_epochs)
  assert np.all(np.isclose(rnn.Wsx, np.array([[0.00856855],
        [0.01936238],
        [0.01382334],
        [0.00771265]])))
  assert np.all(np.isclose(rnn.Wss, np.array([[0.01505222, 0.02059889, 0.01594291, 0.00558242],
        [0.00824307, 0.01741733, 0.01864768, 0.01079751],
        [0.01577334, 0.0124164 , 0.0171516 , 0.0071486 ],
        [0.00722321, 0.00497032, 0.00514737, 0.00979516]])))
  assert np.all(np.isclose(rnn.Wo, np.array([[0.02522786, 0.01744193, 0.01995478, 0.0084726 ]])))
  assert np.all(np.isclose(rnn.Wss0, np.array([[0.01502946],
        [0.01181406],
        [0.02051297],
        [0.00716202]])))
  assert np.all(np.isclose(rnn.Wo0, np.array([[0.42198824]])))
  print("Test passed!")
  return (rnn.Wsx, rnn.Wss, rnn.Wo, rnn.Wss0, rnn.Wo0)

delay_num_test(bptt)

