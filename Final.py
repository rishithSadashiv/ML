#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = 'C:\\Users\\Win\\Desktop\\parkinsons_updrs.csv'


# In[3]:


data = pd.read_csv(path,header = None,names=["subject","age","sex","motor_UPDRS","total_UPDRS","test_time","Jitter(%)","Jitter(Abs)","Jitter:RAP","Jitter:PPQ5","Jitter:DDP","Shimmer","Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","Shimmer:APQ11","Shimmer:DDA","NHR","HNR","RPDE","DFA","PPE"])


# In[4]:


data.head()


# In[5]:


data.drop(['subject','age','sex'],axis = 1, inplace = True)


# In[ ]:





# In[6]:


data.head()


# In[ ]:





# In[7]:


train,test = train_test_split(data,test_size = 0.2)


# In[8]:


columns = data.shape[1]
x_train = train.iloc[:,2:columns]
y_train = train.iloc[:,0:2]
x_test = test.iloc[:,2:columns]
y_test = train.iloc[:,0:2]
y_train = np.matrix(y_train)
x_train = np.matrix(x_train)
y_train = y_train.astype("int")


# In[9]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)


# In[ ]:


lr,y_pred,y_pred.shape


# In[ ]:


count =0
for i,j in zip(y_test,y_pred):
    if(i == j):
        count += 1
count/len(y_pred)*100


# In[ ]:




