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


path = 'C:\\Users\\Win\\Desktop\\titanic.csv'


# In[3]:


data = pd.read_csv(path,header=None,names=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'
])


# In[4]:


y = np.array(data.Survived)


# In[5]:


import seaborn as sb
sb.boxplot(x='Pclass',y='Age',data= data,palette = 'hls')


# In[6]:


def age_approx(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age) == None:
        if(Plcass == 1):
            return 37
        elif(Pclass == 2):
            return 28
        else:
            return 23
    else:
        return Age

data['Age'] = data[['Age','Pclass']].apply(age_approx,axis = 1)


# In[7]:


data.drop('Cabin',axis = 1, inplace = True)


# In[8]:


data.dropna(inplace= True,axis =0)


# In[9]:


data


# In[10]:


data.isnull().sum()


# In[11]:


gender = pd.get_dummies(data['Sex'],drop_first = True)


# In[12]:


embark = pd.get_dummies(data['Embarked'],drop_first = True)


# In[13]:


data.drop(['Sex','Embarked'],axis =1,inplace = True)


# In[14]:


data.drop(['Name','Ticket','PassengerId'],axis =1,inplace = True)


# In[15]:


data.head()


# In[16]:


titanic_dumy = pd.concat([data,gender,embark],axis=1)
titanic_dumy


# In[17]:


titanic,test = train_test_split(titanic_dumy,test_size = 0.3)


# In[18]:


columns = titanic_dumy.shape[1]
x_train = titanic.iloc[:,1:columns]
y_train = titanic.iloc[:,0:1]
x_test = test.iloc[:,1:columns]
y_test = test.iloc[:,0:1]
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[19]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)


# In[20]:


lr,y_pred,y_pred.shape


# In[21]:


count =0
for i,j in zip(y_test,y_pred):
    if(i == j):
        count += 1
count/len(y_pred)*100


# In[ ]:




