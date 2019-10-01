#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = 'C:\\Users\\Win\\Desktop\\ex2data1.csv'


# In[3]:


data = pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])


# In[4]:


data.head()


# In[5]:


positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]


# In[6]:


fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam1'],positive['Exam2'],s=10,c='b',marker='o',label="Admitted")
ax.scatter(negative['Exam1'],negative['Exam2'],s=10,c='r',marker='x',label="Not Admitted")
ax.legend()
ax.set_xlabel("Exam1 Score")
ax.set_ylabel("Exam2 Score")


# In[7]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[8]:


sigmoid(0)


# In[9]:


def cost(theta,x,y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    return np.sum(first-second)/len(x)


# In[10]:


data.insert(0,'Ones',1)
columns = data.shape[1]
x = data.iloc[:,0:columns-1]
y = data.iloc[:,columns-1:columns]
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.zeros(3))


# In[11]:


print(x.shape)
print(y.shape)
print(theta.shape)


# In[12]:


cost(theta,x,y)


# In[13]:


def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    costs = np.zeros(iters)
    for i in range(iters):
        error = sigmoid(X*theta.T) -y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            
        theta = temp
        costs[i] = cost(theta,X,y)
        
    return theta,costs


# In[14]:


g,cost1 = gradientDescent(x,y,theta,0.001,100000)


# In[15]:


g


# In[16]:


cost1


# In[17]:


print(sigmoid(g[0,0]+(g[0,1]*float(input("Enter the Exam1 marks:")))+ (g[0,2]*float(input("Enter the Exam2 marks:")))))


# In[18]:


p = np.array(np.zeros(100))
k=0
for i,j in zip(data.Exam1,data.Exam2):
    if((sigmoid(g[0,0]+(g[0,1]*i)+ (g[0,2]*j)) >= 0.5)):
       p[k]=1
    else:
       p[k]=0
    k+=1
    


# In[19]:


p


# In[20]:


k = 0
c=0
for i in data.Admitted:
    if(p[k] == i):
        c += 1
    k+=1    


# In[21]:


c


# In[ ]:




