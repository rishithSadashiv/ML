#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = 'C:\\Users\\Win\\Desktop\\winequality-white.csv'


# In[3]:


train = pd.read_csv(path,header = None,names=['Fixed Acid','Volatile Acid','Citric Acid','Residual Sugar','Chlorides','Free Sulfur Dioxide','Total Sulfur Dioxide','Density','pH','Sulphates','alcohol','quality'])


# In[4]:


data,test_split = train_test_split(train,test_size = 0.3)


# In[5]:


data,test_split


# In[6]:


data.head()


# In[7]:


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



# In[8]:


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


# In[9]:


data.insert(0,'Ones',1)
columns = data.shape[1]
x = data.iloc[:,0:columns-1]
theta = np.matrix(np.zeros(12))


# In[10]:


p = np.array(np.zeros(3428))
k = 0
for i in data.quality:
    if(i == 4):
        p[k] = 1
    elif(i == 5):
        p[k] = 2
    else:
        p[k] = 3
    k+=1


# In[11]:


x = np.matrix(x.values)


# In[12]:


y = np.matrix(p).T
y.shape,x.shape


# In[13]:


gradi = gradient(theta,x,y,1)


# In[14]:


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


# In[15]:


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


# In[16]:


all_theta = one_vs_all(x,y,3,1)


# In[17]:


y_pred = predict_all(x,all_theta)
correct = [1 if a==b else 0 for (a,b) in zip(y_pred,y)]
accuracy = (sum(map(int,correct))/ float(len(correct)))
print(accuracy*100)


# In[18]:


a = test_split.iloc[0:,:columns-1]


# In[19]:


a.insert(0,'Ones',1)


# In[20]:


all_theta.shape


# In[21]:


a.shape

