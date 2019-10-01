#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = "C:\\Users\\Win\\Desktop\\AutoInsuranceSwedanData.csv"


# In[3]:


data = pd.read_csv(path,header = None, names=['NumberOfClaims','NumberOfAmounts'])


# In[4]:


data


# In[5]:


data.describe()


# In[6]:


data.plot(kind ='scatter',x ='NumberOfClaims', y='NumberOfAmounts', figsize=(12,8))


# In[7]:


def computeCost(x,y,beta):
    inner = np.power((x*beta.T - y),2)
    p = np.sum(inner)/(2*len(x))
    return p


# In[8]:


data.insert(0,'Ones',1)


# In[9]:


data


# In[10]:


columns = data.shape[1]


# In[11]:


x = data.iloc[:,0:columns-1]
y = data.iloc[:,columns-1:columns]


# In[12]:


x = np.matrix(x.values)
y = np.matrix(y.values)


# In[13]:


beta = np.matrix(np.array([0,0]))


# In[14]:


cost = computeCost(x,y,beta)


# In[15]:


cost


# In[16]:


def gradientDescent(X,y,beta,alpha,iters):
    temp = np.matrix(np.zeros(beta.shape))
    parameters = int(beta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = X*beta.T -y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = beta[0,j] - ((alpha/len(X))*np.sum(term))
            
        beta = temp
        cost[i] = computeCost(X,y,beta)
        
    return beta,cost


# In[17]:


g,cost = gradientDescent(x,y,beta,0.001,10000)


# In[18]:


g


# In[19]:


cost


# In[22]:


fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(10000),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# In[23]:


x = np.linspace(data.NumberOfClaims.min(),data.NumberOfClaims.max(),100)


# In[24]:


f = g[0,0] + (g[0,1]*x)


# In[30]:


fig,ax = plt.subplots(figsize=(10,6))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.NumberOfClaims,data.NumberOfAmounts)
ax.legend(loc=4)
ax.set_xlabel("NumberOfClaims")
ax.set_ylabel("NumberOfAmounts")


# In[ ]:




