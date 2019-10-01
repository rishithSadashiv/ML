#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


x = np.array([108,19,13,124,40,57,23,14,45,10])


# In[3]:


y = np.array([392.5,46.2,15.7,422.2,119.4,170.9,56.9,77.5,214,65.3])


# In[4]:


x


# In[5]:


y


# In[6]:


x.reshape(1,10)


# In[7]:


y.reshape(1,10)


# In[8]:


num = np.cov(x,y)


# In[9]:


num


# In[10]:


b1 = num[0,1]/num[0,0]


# In[11]:


b0 = np.mean(y) - b1*np.mean(x)


# In[12]:


b0


# In[13]:


b0+b1*108


# In[ ]:




