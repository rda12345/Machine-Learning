#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW4 part 1
Solution to the initial part of the homework.
"""
import numpy as np

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5



## Q1

def gamma(x,y,th,th0):
    return (y*(th.T@x+th0)/np.sqrt(th.T@th))[0][0]

l_blue = [gamma(data[:,i:(i+1)],labels[0,i:(i+1)],blue_th,blue_th0) for i in range(data.shape[1])]
print('blue: ',l_blue)

print('blue result: ',(sum(l_blue),min(l_blue),max(l_blue)))

l_red = [gamma(data[:,i:(i+1)],labels[0,i:(i+1)],red_th,red_th0) for i in range(data.shape[1])]
print('red result: ',(sum(l_red),min(l_red),max(l_red)))


## Q3

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4


def hinge_loss(gamma,gamma_ref):
        return max([0,1-(gamma/gamma_ref)])


gamma_list = [gamma(data[:,i:(i+1)],labels[0,i:(i+1)],th,th0) for i in range(data.shape[1])]
gamma_ref = np.sqrt(2)/2
result_list = []
for gamma in gamma_list:
    result_list.append(hinge_loss(gamma,gamma_ref))

print('hinge results', result_list)    

## Q4
# 4B [0.001,0]
# 4C [0.03,0.001,0]

## Q6
def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def gd(f, df, x0, step_size_fn, max_iter):
    x = x0
    xs = [x0]
    fs = [f(x0)]
    for t in range(max_iter):
        step_size = step_size_fn(t)
        x = x - step_size*df(x)
        xs.append(x)
        fs.append(f(x))
    return (x,fs,xs)
        

def num_grad(f, delta=0.001):
    def df(x):
      df = np.zeros(x.shape)
      for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += delta
        x_minus[i] -= delta
        df[i] = (f(x_plus)-f(x_minus))/(2*delta)
      return df
    return df    

# x = cv([0.])
# ng = num_grad(f1)(x)
# print('check',num_grad(f1)(x))

# ans=(num_grad(f1)(x).tolist(), x.tolist())
# print(ans)



def minimize(f, x0, step_size_fn, max_iter):
    df = num_grad(f,delta = 0.001)
    x = x0
    xs = [x0]
    fs = [f(x0)]
    for t in range(max_iter):
        step_size = step_size_fn(t)
        x = x - step_size*df(x)
        xs.append(x)
        fs.append(f(x))
    return (x,fs,xs)

## Q7.1

def hinge(v):
    return np.where(v < 1,1-v,np.zeros(v.shape))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y*(np.dot(th.T,x)+th0))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    ans = np.sum(hinge_loss(x,y,th,th0))+lam*np.linalg.norm(th)**2
    return ans


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator

ans = svm_obj(x_1, y_1, th1, th1_0, .1)
#print(ans)
# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
#print(ans)

## Q7.2

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v < 1,-np.ones(v.shape),np.zeros(v.shape))

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T,x)+th0))*y*x

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T,x)+th0))*y


# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis=1, keepdims=True)+2*lam*th

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0),keepdims = True)

# Returns the full gradient as a single vector
def svm_obj_grad(X, y, th, th0, lam):
    temp_th = d_svm_obj_th(X, y, th, th0, lam)
    temp_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack((temp_th,temp_th0))


# Test 1

X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])


## Q7.3
def batch_svm_min(data, labels, lam):
    ''' Evaluates the minimum of the svm function employing gradient descent
        
        Input
            data: (d,n) array
            labels: (1,n) array
            lam: scalar
            
        Parameters:
            max_iter: int, maxium number of iteractions, set as 10
            th: (d,1) array, forms the seperator
            th0: (1,1) array, forms the seperator
        
        Returns
            tuple (x,fs,xs), where 
            x: (d+1,1) array, containing the seperator (th,th0) (th0 is the last element of the vector)
            fs: list, values of the svm function for 
            
    '''
    d = data.shape[0]
    
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    
    def f(x):
        return svm_obj(data,labels,x[:-1,0:1],x[d:d+1,0:1],lam)    
    
    def df(x):
        return svm_obj_grad(data,labels,x[:-1,0:1],x[d:d+1,0:1],lam)
    
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))
    x0 = np.vstack((th,th0))
    max_iter = 10
    
    return gd(f, df, x0, svm_min_step_size_fn, max_iter)

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]


def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

x_1, y_1 = super_simple_separable()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))

x_1, y_1 = separable_medium()
ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))
    
    
    