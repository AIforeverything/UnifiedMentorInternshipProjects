#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


# In[2]:


# making a path to get the modules from categorical library
import os
os.chdir('..')


# In[3]:


from categorical.categorical_model import categorical_Model


# In[4]:


categorical_Model.extractingZipFile("./Liver Cirrhosis Stage Detection/liver_cirrhosis_stage.zip",'./Liver Cirrhosis Stage Detection/')


# In[5]:


df=pd.read_csv("./Liver Cirrhosis Stage Detection/liver_cirrhosis_stage/liver_cirrhosis.csv")
df.head()


# ## EDA

# In[6]:


df.info()


# ### Removing duplicates

# In[7]:


categorical_Model.checkDuplicates(df)


# ### Checking and Removing missing values

# In[8]:


# Checking missing values for each column
print(categorical_Model.missing_columns(df))
#checking missing values of all columns
print(categorical_Model.missing_columns_total(df))
df.dropna(inplace=True)
df.head()


# In[9]:


def showingUnique(x):
    return x.unique()


# In[10]:


c=list(df.columns)
for i in c:
    if df.dtypes[i]=='object':
        print(i,showingUnique(df[i]))


# ### Splitting into X and y before standardize X

# In[11]:


X,y=categorical_Model.splitDataIntoXy(df,"Stage")


# ### Standardizing the numerical columns of X

# In[12]:


categorical_Model.standardizeNonCategoricalColumns(X)


# ### converting categorical columns to numerical of X

# In[13]:


categorical_Model.checkCategoricalColumnsAndReplacingWithLE(X)


# In[14]:


X.info()


# In[15]:


y.info()


# In[16]:


y.unique()


# In[17]:


X_train, X_test, y_train, y_test=categorical_Model.splitData(X,y)


# In[18]:


categorical_Model.trainModel(X_train, X_test, y_train, y_test)


# 
