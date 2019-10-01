#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = "C:\\Users\\Win\\Desktop\\data1.csv"


# In[3]:


data = pd.read_csv(path,header = None,names = ['height','weight'])


# In[5]:


data.head()


# In[6]:


data.plot(kind = 'scatter',x = 'height',y = 'weight',figsize = (12,8))


# In[ ]:





# In[ ]:




