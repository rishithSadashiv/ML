#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


degree = 6
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
data.drop('Ones',axis =1,inplace = True)


# In[10]:


data_Train,data_test = train_test_split(data,test_size=0.2)


# In[11]:


data_Train.shape,data_test.shape


# In[12]:


columns = data.shape[1]
x_train = data_Train.iloc[:,1:columns]
y_train = np.array(data_Train.iloc[:,0:1])
x_test = data_test.iloc[:,1:columns]
y_test = np.array(data_test.iloc[:,0:1])


# In[13]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[14]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)


# In[15]:


y_pred


# In[16]:


count =0
for i,j in zip(y_test,y_pred):
    if(i == j):
        count += 1
count/len(y_pred)*100

