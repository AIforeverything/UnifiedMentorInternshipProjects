#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# making a path to get the modules from categorical library
import os
os.chdir('..')


# In[5]:


from categorical.categorical_model import categorical_Model


# In[6]:


categorical_Model.extractingZipFile("./DetectLungCancerUsingPatientDiagnosisData/lung_cancer.zip",'./DetectLungCancerUsingPatientDiagnosisData/')


# In[7]:


df=pd.read_csv("./DetectLungCancerUsingPatientDiagnosisData/Lung Cancer/dataset_med.csv")
df.head()


# ## EDA

# ## Following step by stem and analysing and cleaning columns
# #### Step-1 : df["country] is not useful for model buiding so dropping it

# In[8]:


df.drop(['country'],axis=1,inplace=True)
df.head()


# In[9]:


df.info()


# ### Feature Engineering on the columns diagnosis_date and end_treatment_date

# In[10]:


df["end_treatment_date"]=pd.to_datetime(df["end_treatment_date"])
df["diagnosis_date"]=pd.to_datetime(df["diagnosis_date"])
df.info()


# In[11]:


df["treatment_duration"]=df["end_treatment_date"]-df["diagnosis_date"]
df.head()


# ### extracting days from the duration and making into a fraction of a year

# In[12]:


df["treatment_duration"]=df["treatment_duration"].astype(str)
df["treatment_duration"]=df["treatment_duration"].str.extract(r"(\d+)").astype(int)
df.head()


# In[13]:


df["treatment_duration_scaled"]=df["treatment_duration"]/(365.0)
df.head()


# In[14]:


df.drop(["diagnosis_date","end_treatment_date","treatment_duration"],axis=1,inplace=True)
df.head()


# In[15]:


df.info()


# In[16]:


def showingUnique(x):
    return x.unique()


# In[17]:


c=list(df.columns)
for i in c:
    if df.dtypes[i]=='object':
        print(i,showingUnique(df[i]))


# ### converting categorical columns to numerical

# In[18]:


categorical_Model.checkCategoricalColumns(df)


# #### Removing Id column

# In[19]:


df.drop(["id"],axis=1,inplace=True)
df.head()


# ### Removing duplicates

# In[20]:


categorical_Model.checkDuplicates(df)


# ### Removing missing values

# In[21]:


# Checking missing values for each column
print(categorical_Model.missing_columns(df))


# In[22]:


#checking missing values of all columns
print(categorical_Model.missing_columns_total(df))


# In[23]:


df.dropna(inplace=True)


# In[24]:


df.head()


# In[25]:


from sklearn.preprocessing import MinMaxScaler
minMax=MinMaxScaler()
df["age_scaled"]=minMax.fit_transform(df[["age"]])
df["bmi_scaled"]=minMax.fit_transform(df[["bmi"]])
df["cholesterol_level_scaled"]=minMax.fit_transform(df[["cholesterol_level"]])
df.head()


# In[26]:


df.drop(["age","bmi","cholesterol_level"],axis=1,inplace=True)


# In[27]:


categorical_Model.printCorrelationMatrix(df,"survived")


# In[28]:


X_train, X_test, y_train, y_test=categorical_Model.splitData(df,"survived")


# In[29]:


categorical_Model.trainModel(X_train, X_test, y_train, y_test)


# 
