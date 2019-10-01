#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
path = "C:\\Users\\Karthik S\\Desktop\\dataset.csv"
data = pd.read_csv(path, header = None)


# In[2]:


data.head()


# In[3]:


def split_train_test(datasets, percentage):
    total_row,total_col = datasets.shape
    total_row_data = total_row*percentage//100
    train = datasets[:][0:(total_row-total_row_data)] 
    test = datasets[:][(total_row-total_row_data):]
    return train,test


# In[6]:


def extract_spam_notspam(datasets):
    j=0
    k=0
    row,col = datasets.shape
    spam = np.zeros((row,col))
    notspam = np.zeros((row,col))
    for i in range(0,row):
        if(datasets[57][i] == 1):
            spam[j] = datasets[:][i:i+1]
            j = j+1
        else:
            notspam[k] = datasets[:][i:i+1]
            k = k+1
            
    return spam,notspam


# In[7]:


[spam,notspam] = extract_spam_notspam(data)


# In[8]:


spam.head()


# In[9]:


spam


# In[10]:


spam.shape


# In[11]:


data.shape


# In[12]:


data[57]


# In[13]:


spam.shape


# In[18]:


x = spam[~np.all(spam==0,axis=1)]


# In[19]:


x.shape


# In[ ]:




