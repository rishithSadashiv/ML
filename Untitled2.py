#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[ ]:


path = "C:\\Users\\Win\\Desktop\\dataset.csv"


# In[ ]:


data = pd.read_csv(path, header=None)


# In[ ]:


data.head()


# In[ ]:


spam = data[data[57].isin([1])]
notspam = data[data[57].isin([0])]


# In[ ]:





# In[ ]:




