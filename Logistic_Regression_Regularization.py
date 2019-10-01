#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = "C:\\Users\\Win\\Desktop\\ex2data2.csv"


# In[3]:


data = pd.read_csv(path,header=None,names=['Test1','Test2','Accepted'])


# In[4]:


data.head()


# In[5]:


positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]


# In[6]:


fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test1'],positive['Test2'],s=50,c='b',marker='o',label="Accepted")
ax.scatter(negative['Test1'],negative['Test2'],s=50,c='r',marker='x',label="Not Accepted")
ax.legend()
ax.set_xlabel("Test1 Score")
ax.set_ylabel("Test2 Score")


# In[7]:


degree = 10
x1 = data['Test1']
x2 = data['Test2']
data.insert(3,'Ones',1)
for i in range(1,degree):
    for j in range(0,i+1):
        data['F'+str(i)+str(j)] = np.power(x1,i-j)*np.power(x2,j)


# In[8]:


data.head()


# In[9]:


data.drop('Test1', axis =1, inplace = True)
data.drop('Test2', axis =1, inplace = True)


# In[10]:


data.head()


# In[11]:


def sigmoid(z):
    return 1/(1+np.exp(-z))

def costReg(theta,x,y,lambda1):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    reg = (lambda1/(2*len(x)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/(len(x)) + reg


# In[12]:


def gradientReg(theta,x,y,lambda1):
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


# In[13]:


columns = data.shape[1]
x = data.iloc[:,1:columns]
y = data.iloc[:,0:1]
x = np.array(x.values)
y = np.array(y.values)
theta = np.zeros(55)
lambda1 = 1
costReg(theta,x,y,lambda1)


# In[14]:


import scipy.optimize as opt


# In[15]:


result = opt.fmin_tnc(func = costReg,x0=theta,fprime=gradientReg,args=(x,y,10))


# In[16]:


result


# In[17]:


p = np.array(np.zeros(118))
result = np.matrix(result)        


# In[18]:


result


# In[19]:


a = result[0,0]


# In[20]:


a


# In[21]:


a.shape


# In[22]:


x = np.matrix(x)
x.shape


# In[23]:


p = np.array(np.zeros(118))
k =0
b = x*a
for i in b:
    if(sigmoid(i) > 0.5):
        p[k] = 1
    else:
        p[k] = 0
    k+=1


# In[24]:


p


# In[25]:


l = 0
c=0
for i in data.Accepted:
    if(p[l] == i):
        c += 1
    l+=1 
print(c/118)


# In[ ]:





# In[ ]:




