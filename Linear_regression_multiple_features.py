#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = "C:\\Users\\Win\\Desktop\\ex2.csv"


# In[3]:


data = pd.read_csv(path,header = None, names=['Area','NoOfRooms','Price'])


# In[4]:


data


# In[5]:


data.describe()


# In[6]:


data = (data - data.mean())/data.std()


# In[7]:


data.insert(0,'Ones',1)


# In[8]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
x=data.Area
y=data.NoOfRooms
z=data.Price
ax.scatter(x,y,z,c='r',marker='o')
ax.set_xlabel('AREA')
ax.set_ylabel('NO OF ROOMS')
ax.set_zlabel('PRICE')


# In[9]:



def computeCost(x,y,theta):
    inner = np.power((x*theta.T-y),2)
    p = np.sum(inner)/(2*len(x))
    return p


# In[10]:


data.head()


# In[11]:


columns = data.shape[1]


# In[12]:


x = data.iloc[:,0:columns-1]


# In[13]:


y = data.iloc[:,columns-1:columns]


# In[14]:


x.values


# In[15]:


y.values


# In[16]:


x = np.matrix(x.values)


# In[17]:


y = np.matrix(y.values)


# In[18]:


theta = np.matrix(np.array([0,0,0]))


# In[19]:


cost = computeCost(x,y,theta)


# In[20]:


cost


# In[21]:


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


# In[22]:


g,cost = gradientDescent(x,y,theta,0.001,10000)


# In[23]:


g


# In[24]:


cost


# In[ ]:





# In[ ]:




