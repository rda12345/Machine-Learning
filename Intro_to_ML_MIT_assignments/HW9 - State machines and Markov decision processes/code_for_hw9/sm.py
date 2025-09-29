from util import *
import numpy as np

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        output_list = []
        st = self.start_state
        for xt in input_seq:
            st = self.transition_fn(st,xt)
            yt = self.output_fn(st)
            output_list.append(yt)
        return output_list
            


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s
    
## Accumulator check

# sm = Accumulator()
# expected_output = [-1, 1, 4, 2, 7, 13]
# output = sm.transduce([-1, 2, 3, -2, 5, 6])
# print(f'Accumulator test:\n result - {output}, expected - {expected_output} \n')



class Binary_Addition(SM):
    start_state = (0,0)

    def transition_fn(self, s, x):
        return ((x[0]+x[1]+s[1])%2, (x[0]+x[1]+s[1])//2)

    def output_fn(self, s):
        return s[0]
    
## Binary addition check

# sm = Binary_Addition()
# expected_output = [0, 0, 1]
# output = sm.transduce([(1, 1), (1, 0), (0, 0)])
# print(f'Transduce test:\n result - {output}, expected - {expected_output} \n')


class Reverser(SM):
    # The state has two types of parameters, the first is a list (naturally one would use a stack here)
    # which stores the elements of sequence1 and the second is mode of operation. The 
    # later is parameterized by a boolian. If True the function stores the sequence
    # and returns None. If False the transition function removes the elemens from the 
    # stack and returns them.
    start_state = ([],True)
    
    def transition_fn(self, s, x):
        # If the content of the list is end, switch the mode of operation of the function, and switch again 
        # once the list is empty
        mode = s[1]
        if mode:
            if x == 'end':
                st = (s[0],False)
                return st
            s[0].append(x)
            st = (s[0],True)    
        # When mode == False
        elif  len(s[0])>0:
            s[0].pop()
            st = (s[0],False)
        else:
            st = ([],True)
        return st   
            
    def output_fn(self, s):
        mode = s[1]
        if  mode or len(s[0]) == 0:
            return None
        return s[0][-1]
    
## Reverse check

# sm = Reverser()
# expected_output = [None, None, None, 'bar', ' ', 'foo', None, None, None]
# output = sm.transduce(['foo', ' ', 'bar'] + ['end'] + list(range(5)))
# print(f'Reverser test:\n result - {output},\n expected - {expected_output}\n')

class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx = Wsx 
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2
        self.n = self.Wss.shape[1]
        self.start_state = np.zeros((self.n,1))

        
    def transition_fn(self, s, i):
        st = self.f1(np.dot(self.Wss,s) + np.dot(self.Wsx,i) + self.Wss_0)
        return st
    
    def output_fn(self, s):
        yt = self.f2(np.dot(self.Wo,s)+self.Wo_0)
        return yt
    


    
'''
Exercise solutions:
    
## Q1.5

Wsx = np.ones((1,1))   
Wss = np.ones((1,1))
Wo = 100*np.ones((1,1))  
Wss_0 = np.zeros((1,1))
Wo_0 =  np.zeros((1,1))
f1 = lambda x : x   # Your code here, e.g. lambda x : x
f2 = lambda x : np.tanh(x) 
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)


## Q1.6

Wsx = np.array([[1,0,0]]).T   
Wss = np.array([[0,0,0],[1,0,0],[0,1,0]])
Wo = np.array([[1,-2,3]])  
Wss_0 = np.zeros((1,1))
Wo_0 =  np.zeros((1,1))
f1 = lambda x : x   # Your code here, e.g. lambda x : x
f2 = lambda x : x 
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)


## Q1.7 

# The transition matrices of actions 'b' and 'c'
Tb = np.array([[0.0, 0.9, 0.1, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])
Tc = np.array([[0.0, 0.1, 0.9, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])

R = np.array([[0,1,0,2]]).T

# Choosing c with horizon = 2
V2 = R + np.sum(Tc@R,axis = 1, keepdims = True)
V3 = V2 + np.sum(Tc@V2,axis = 1, keepdims = True)
V4 = V3 + np.sum(Tc@V3,axis = 1,keepdims = True)
print(f'Choosing c policy with horizon 2: {round(V3.T.tolist()[0][0],3)}')
print(f'Choosing c policy with horizon 3: {round(V4.T.tolist()[0][0],3)}')


# Choosing c with horizon = 2
V2 = R + np.sum(Tb@R,axis = 1, keepdims = True)
V3 = V2 + np.sum(Tb@V2,axis = 1, keepdims = True)
V4 = V3 + np.sum(Tb@V3,axis = 1,keepdims = True)
print(f'Choosing b policy: {round(V3.T.tolist()[0][0],3)}')
print(f'Choosing b policy with horizon 3: {round(V4.T.tolist()[0][0],3)}')

## Q2 

Tb = np.array([[0.0, 0.9, 0.1, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])
Tc = np.array([[0.0, 0.1, 0.9, 0.0],[0.9, 0.1, 0.0, 0.0],[0.0, 0.0, 0.1, 0.9],[0.9, 0.0, 0.0, 0.1]])
gamma = 0.9
print(f'Q2.2B: {Tc*gamma}')

A = gamma*Tc-np.eye(4)
R = np.array([[0,1,0,2]]).T
b = -R
v = np.linalg.solve(A,b)  # A v = b
print(f'Q2.2C: {v.T.tolist()[0]}')

## Q3 

# second iteration Q(s,'b')
ans1 = R + gamma*Tb@R
print(f'Q(s,"b"): {ans1.T.tolist()[0]}')
ans2 = R + gamma*Tc@R
print(f'Q(s,"c"): {ans2.T.tolist()[0]}')

# second iteration Q(s,'b')
ans3 = R + gamma*Tb@ans1
print(f'Q(s,"b"): {ans3.T.tolist()[0]}')
ans4 = R + gamma*Tc@ans1
print(f'Q(s,"c"): {ans4.T.tolist()[0]}')
'''