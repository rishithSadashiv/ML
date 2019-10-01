#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = loadmat('C:\\Users\\Win\\Desktop\\ex3data1.mat')


# In[3]:


data


# In[4]:


data['X'].shape,data['y'].shape


# In[5]:


np.unique(data['y'])


# In[6]:


def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(theta,x,y,lambda1):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    reg = (lambda1/(2*len(x)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/(len(x)) + reg


# In[7]:


def gradient(theta,x,y,lambda1):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(x*theta.T)-y
    for i in range(parameters):
        term = np.multiply(error,x[:,i])
        if (i==0):
            grad[i] = np.sum(term)/len(x)
        else:
            grad[i] = (np.sum(term)/len(x)) + (((lambda1)/len(x))*theta[:,i])
        
    return grad


# In[8]:


from scipy.optimize import minimize
def one_vs_all(x,y,num_labels,learning_rate):
    rows = x.shape[0]
    params = x.shape[1]
    all_theta = np.zeros((num_labels,params+1))
    x = np.insert(x,0,values = np.ones(rows),axis = 1)
    for i in range(1,num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i,(rows,1))
        fmin = minimize(fun = cost,x0= theta,args=(x,y_i,learning_rate),method = 'TNC',jac = gradient)
        all_theta[i-1,:] = fmin.x
        
    return all_theta


# In[9]:


def predict_all(x,all_theta):
    rows = x.shape[0]
    params = x.shape[1]
    num_labels = all_theta.shape[0]
    x = np.insert(x,0,values=np.ones(rows),axis = 1)
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)
    h = sigmoid(x*all_theta.T)
    h_argmax = np.argmax(h,axis=1)
    h_argmax = h_argmax+1
    return h_argmax


# In[10]:


all_theta = one_vs_all(data['X'],data['y'],10,1)


# In[11]:


y_pred = predict_all(data['X'],all_theta)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred,data['y'])]
accuracy = (sum(map(int,correct))/ float(len(correct)))
print(accuracy*100)


# In[43]:


x = data['X'][1246]
x = np.insert(x,0,1)
x = np.matrix(x)


# In[44]:


y_pre = sigmoid(x*all_theta.T)


# In[45]:


y_pre


# In[46]:


np.max(y_pre)


# In[47]:


data['y'][1246]


# In[ ]:




