#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = "C:\\Users\\Win\\Desktop\\ex1data1.csv"


# In[3]:


data = pd.read_csv(path,header = None, names=['Populations','Profit'])


# In[4]:


data


# In[5]:


data.describe()


# In[6]:


data.plot(kind ='scatter',x ='Populations', y='Profit', figsize=(12,8))


# In[7]:


def computeCost(X,y,theta):
    inner = np.power((X*theta.T -y),2)
    p = np.sum(inner)/(2*len(x))
    return p


# In[8]:


data.insert(0,'Ones',1)


# In[9]:


data


# In[10]:


columns = data.shape[1]


# In[11]:


columns


# In[12]:


x = data.iloc[:,0:columns-1]
y = data.iloc[:,columns-1:columns]


# In[13]:


y.values


# In[14]:


type(x)


# In[15]:


x.values


# In[16]:


x = np.matrix(x.values)
y = np.matrix(y.values)


# In[17]:


type(x)


# In[18]:


theta = np.matrix(np.array([0,0]))


# In[19]:


theta.shape


# In[20]:


cost = computeCost(x,y,theta)


# In[21]:


cost


# In[22]:


def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = X*theta.T -y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X,y,theta)
        
    return theta,cost


# In[23]:


g,cost = gradientDescent(x,y,theta,0.01,1000)


# In[24]:


g


# In[25]:


cost


# In[26]:


x = np.linspace(data.Populations.min(),data.Populations.max(),100)


# In[27]:


f = g[0,0] + (g[0,1]*x)


# In[28]:


fig,ax = plt.subplots(figsize=(10,6))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Populations,data.Profit)
ax.legend(loc=4)
ax.set_xlabel("Populations")
ax.set_ylabel("Profit")


# In[ ]:




