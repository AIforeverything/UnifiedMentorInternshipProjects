#!/usr/bin/env python
# coding: utf-8

# # Objective
# ## Build a system that can predict if a Thyroid Cancer survivor can relapse(his or her cancer reoccurs)
# ### Dataset
# #### This dataset contains data about thyroid checkups for people with a diagnosis and is a comprehensive collection of patient information, specifically focused on individuals diagnosed with cancer

# ## Step-1: Common virtual environment was created and activated: myenv
# #### pip install virtualenv
# #### virtualenv myenv
# #### .\myenv\Scripts\activate.ps1

# ## Installing required libraries

# In[1]:


# %pip install -r requirements.txt


# ## Step-2: Importing required libraries

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import zipfile
import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ## Step-3: Data extraction from zipfile

# In[4]:


def extractingZipFile(zipFilePath, extractTo):
    """
    Extracts the contents of a zip file to a specified directory.
    
    Parameters:
    zipFilePath (str): The path to the zip file.
    extractTo (str): The directory to extract the contents to.
    """
    with zipfile.ZipFile(zipFilePath, 'r') as zip_ref:
        zip_ref.extractall(extractTo)
extractingZipFile('thyroid_cancer.zip', 'data')        


# ## Step-4: Importing data into a dataframe

# In[13]:


def readingData(path):
    """
    Reads the data from a CSV file and returns it as a pandas DataFrame.
    Parameters:
    path (str): The path to the CSV file.
    Returns:
    pd.DataFrame: The data as a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df
df=readingData("data/thyroid_cancer/dataset.csv")
df.head()


# ## Step-4: EDA (Exploratory Data Analysis)

# In[14]:


df.info()


# In[15]:


df.describe()


# ## Step-4(a): Checking missing values

# In[16]:


def checkMissingValues(df):
    """
    Checks for missing values in the DataFrame 
    Parameters:
    df (pd.DataFrame): The DataFrame to check for missing values.
    Returns:
    missing values
    """
    return df.isnull().sum()

missing_values = checkMissingValues(df)
missing_values


# #### No missing values were found

# ## Step-4(b): Removing duplicates

# In[17]:


## function to check for duplicates and remove dupliates
def checkDuplicates(df):
    """
    Checks for duplicate rows in the DataFrame and removes them.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check for duplicates.
    
    Returns:
    pd.DataFrame: The DataFrame with duplicates removed.
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")
    else:
        print("No duplicate rows found.")
    return df
df = checkDuplicates(df)
df.head()


# In[18]:


## function to check categorical columns and replacing them with numerical values
def checkCategoricalColumns(df):
    """
    Checks for categorical columns in the DataFrame and replaces them with numerical values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check for categorical columns.
    
    Returns:
    pd.DataFrame: The DataFrame with categorical columns replaced with numerical values.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {categorical_columns}")

    for col in categorical_columns:
        print(f"col.unique(): {df[col].unique()}")
        print(f"col.value_counts(): {df[col].value_counts()}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
df = checkCategoricalColumns(df)
df.head()


# In[19]:


df.info()


# In[20]:


df.corr()["Recurred"].sort_values(ascending=False)


# ## Step-5: model building

# In[21]:


X=df.drop(columns=['Recurred'])
y=df['Recurred']
X.head()


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


## different models training using gridserachCV and evaluation
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a machine learning model using GridSearchCV.
    
    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to train.
    param_grid (dict): The parameter grid for GridSearchCV.
    X_train (pd.DataFrame): The training data.
    y_train (pd.Series): The training labels.
    X_test (pd.DataFrame): The testing data.
    y_test (pd.Series): The testing labels.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return best_model
# Logistic Calssifier
logistic_model = LogisticRegressionCV(max_iter=1000)
logistic_param_grid = {
    'Cs': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
logistic_best_model = train_and_evaluate_model(logistic_model, logistic_param_grid, X_train, y_train, X_test, y_test)
# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_best_model = train_and_evaluate_model(rf_model, rf_param_grid, X_train, y_train, X_test, y_test)
# XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_best_model = train_and_evaluate_model(xgb_model, xgb_param_grid, X_train, y_train, X_test, y_test)
# Save the best model
def save_model(model, model_name):
    """
    Saves the trained model to a file.
    
    Parameters:
    model (sklearn.base.BaseEstimator): The trained model to save.
    model_name (str): The name of the model file.
    """
    joblib.dump(model, model_name)

save_model(logistic_best_model, 'logistic_model.pkl')
save_model(rf_best_model, 'rf_model.pkl')
save_model(xgb_best_model, 'xgb_model.pkl')

## function to print the model accuracy
def print_model_accuracy(model, X_test, y_test):
    """
    Prints the accuracy of the model on the test data.
    
    Parameters:
    model (sklearn.base.BaseEstimator): The trained model to evaluate.
    X_test (pd.DataFrame): The testing data.
    y_test (pd.Series): The testing labels.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")




# In[27]:


print("LogisticRegressionCV_best_model: ")
print_model_accuracy(logistic_best_model, X_test, y_test)
print("RandomForestClassifier_best_model: ")
print_model_accuracy(rf_best_model, X_test, y_test)
print("XGBClassifier_best_model: ")
print_model_accuracy(xgb_best_model, X_test, y_test)


# In[ ]:


# loading the best model and checking precision,recall,f1-score, accuracy
def load_model(model_name):
    """
    Loads a trained model from a file.
    
    Parameters:
    model_name (str): The name of the model file.
    
    Returns:
    sklearn.base.BaseEstimator: The loaded model.
    """
    return joblib.load(model_name)
logistic_model = load_model('logistic_model.pkl')
rf_model = load_model('rf_model.pkl')
xgb_model = load_model('xgb_model.pkl')


# # Step-6:  RandomForestClassifier has maximum
# Accuracy: 0.958904109589041
#               precision    recall  f1-score   support
# 
#            0       0.96      0.98      0.97        51
#            1       0.95      0.91      0.93        22
# 
#     accuracy                           0.96        73
#    macro avg       0.96      0.94      0.95        73
# weighted avg       0.96      0.96      0.96        73
