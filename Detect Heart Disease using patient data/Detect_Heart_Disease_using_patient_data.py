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


categorical_Model.extractingZipFile("./Detect Heart Disease using patient data/heart_disease.zip",'./Detect Heart Disease using patient data/')


# In[5]:


df=pd.read_csv("./Detect Heart Disease using patient data/Heart Disease/dataset.csv")
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


X,y=categorical_Model.splitDataIntoXy(df,"target")


# In[12]:


df["target"].unique()


# ### Standardizing the numerical columns of X

# In[13]:


categorical_Model.standardizeNonCategoricalColumns(X)


# ### converting categorical columns to numerical of X

# In[14]:


categorical_Model.checkCategoricalColumnsAndReplacingWithLE(X)


# In[15]:


X.info()


# In[16]:


y.info()


# In[17]:


y.unique()


# In[18]:


X_train, X_test, y_train, y_test=categorical_Model.splitData(X,y)


# In[19]:


categorical_Model.trainModel(X_train, X_test, y_train, y_test)


# 
