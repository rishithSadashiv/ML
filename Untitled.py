#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


a = np.array([[1,2],[1.2],[1,2]])


# In[4]:


a


# In[6]:


print(np.arange(10000).reshape(100,100))


# In[7]:


a = np.arange(12).reshape(3,4)


# In[11]:


b = np.arange(2,0)


# In[14]:


a+b


# In[15]:


b


# In[16]:


a = np.arange(20).reshape(5,4)


# In[17]:


b = np.arange(4).reshape(1,4)


# In[18]:


a+b


# In[19]:


a-b


# In[20]:


a*b


# In[22]:


a = np.arange(48).reshape(8,1,6,1)


# In[23]:


b = np.arange(35).reshape(7,1,5)


# In[24]:


a+b


# In[25]:


(a+b).shape


# In[27]:


a = np.arange(12).reshape(4,3)


# In[28]:


a.T


# In[29]:


a.shape


# In[30]:


print(a.sum(axis = 0))


# In[31]:


a.sum(axis=1)


# In[32]:


a = np.arange(4).reshape(1,4)


# In[33]:


np.exp(a)


# In[34]:


np.sqrt(a)


# In[35]:


b = np.floor(10*np.random.random((2,2)))


# In[37]:


a = np.array(([2,5],[1,0]))


# In[38]:


a


# In[39]:


a.shape


# In[40]:


np.hstack((a,b))


# In[41]:


np.vstack((a,b))


# In[42]:


a.min()


# In[43]:


a.max()


# In[44]:


a.sum()


# In[45]:


a**2


# In[52]:


a=np.array(([0,2],[1,1],[2,0]))


# In[53]:


a


# In[54]:


a.shape


# In[55]:


a = [-2.1,-1,4.3]


# In[56]:


b=[3,1.1,0.12]


# In[57]:


x = np.stack((a,b),axis = 0)


# In[58]:


x


# In[69]:


a=np.array(([0,2],[1,1],[2,0]))


# In[70]:


b = np.array(([2,0],[1,1],[0,2]))


# In[76]:


x = np.stack((a,b),axis = 0)


# In[77]:


x


# In[75]:


x.shape


# In[78]:


a = [2,1,0]


# In[79]:


b= [0,1,2]


# In[80]:


x = np.stack((a,b),axis = 0)


# In[81]:


x


# In[82]:


np.cov(x)


# In[83]:


a = [-2.1,-1,4.3]


# In[84]:


b=[3,1.1,0.12]


# In[85]:


x = np.stack((a,b),axis = 0)


# In[86]:


x


# In[87]:


np.cov(x)


# In[2]:


import numpy as np


# In[3]:


a = [-2.1,-1,4.3]


# In[4]:


b=[3,1.1,0.12]


# In[5]:


np.cov(a,b)


# In[6]:


x = np.cov(a,b)


# In[7]:


y = np.var(a)


# In[8]:


x/y


# In[ ]:




