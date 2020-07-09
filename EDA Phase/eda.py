# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:58:07 2020

@author: Aditi Jain
"""

pwd
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_excel('alarm_files.xlsx')

#Get the rows that contains NULL
dataset.isnull().sum()

X = dataset.iloc[:,[2, 3, 4]].values     #input variables
y = dataset.iloc[:, -1].values        #target variable


dataset.columns


#Encoding categorical data to create the dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()


#Label Encoding

X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])





#Applying One Hot Encoding to Role Column
hotencoder = OneHotEncoder(categorical_features=[0])
X = hotencoder.fit_transform(X).toarray()

#Applying One Hot Encoding to Component Accessed
hotencoder = OneHotEncoder(categorical_features=[3])
X = hotencoder.fit_transform(X).toarray()

#Applying One Hot Encoding to Request Type
hotencoder = OneHotEncoder(categorical_features=[8])
X = hotencoder.fit_transform(X).toarray()







