#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exersice 10
"""

# Q - learning algorithm
states = [0,1,2,3]
actions = ['b','c']
gamma = 0.9     # Discount factor
alpha = 0.5

Q = {}

for s in states:
    for a in actions:
        Q[(s,a)] = 0

experience = [(0, 'b', 0), #t = 0
              (2, 'b', 0),
              (3, 'b', 2),
              (0, 'b', 0), #t = 3
              (2, 'b', 0),
              (3, 'c', 2),
              (0, 'c', 0), #t = 6
              (1, 'b', 1),
              (0, 'b', 0),
              (2, 'c', 0), #t = 9
              (3, 'c', 2),
              (0, 'c', 0),
              (1, 'c', 1), #t = 12
              (0, 'c', 0),
              (2, 'b', 0),
              (3, 'b', 2), #t = 15
              (0, 'b', 0),
              (2, 'c', 0),
              (3, '', 0), #t = 18
              ]

def Q_cal(experience,alpha,gamma):
    l = []
    for ind, tup in enumerate(experience[:-1]):
        s, a, r = tup
        stag = experience[ind+1][0]
        Q[(s,a)] = (1-alpha)*Q[(s,a)]+alpha*(r+gamma*max([Q[stag,atag] for atag in actions ]))
        l.append(round(Q[(s,a)],3))
    return l
tup0 = (0,'b',0)
tup1 = (2,'b',0)
tup2 = (3,'b',2)

print(f'Q(s,a) values for increasing time steps: \n {Q_cal(experience,alpha,gamma)}')

