# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:18:55 2020

@author: Aditi Jain
"""

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

pwd

dataset = pd.read_csv('company_data.csv')

#Importing stastmodels.formula to fit the regression model
import statsmodels.api as sm

X = dataset.iloc[:, : -1].values     #input variables
y = dataset.iloc[:, 4].values        #target variable

#Encoding categorical data to create the dummy variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() 

#Adding intercept column to X
X = np.append(arr = np.ones((71,1)).astype(int), values = X, axis = 1)

#Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting the first regression model  #Backward Elimination Method
X_fitted = X_train[:,:]
reg_1 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_1.summary()

#Fitting the second regression model by removing highest p-value  #Backward Elimination Method
X_fitted = X_train[:,[0,1,3,4,5,6,7]]
reg_2 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_2.summary()

#Fitting the third regression model by removing highest p-value  #Backward Elimination Method
X_fitted = X_train[:,[0,3,4,5,6,7]]
reg_3 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_3.summary()

#Fitting the forth regression model by removing highest p-value  #Backward Elimination Method
X_fitted = X_train[:,[0,3,4,5,7]]
reg_4 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_4.summary()


#Fitting the fifth regression model by removing highest p-value  #Backward Elimination Method
X_fitted = X_train[:,[0,3,4,5]]
reg_5 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_5.summary()

#At fifth regression it gave powerful regression model as Adjunct R-Squared value is powerful

#Fitting the sixth regression model by removing highest p-value  #Backward Elimination Method
X_fitted = X_train[:,[0,3,5]]
reg_6 = sm.regression.linear_model.OLS(endog = y_train, exog = X_fitted).fit()
reg_6.summary()


X_for_test = X_train[:, [0, 3, 4, 5]]
y_predicted = reg_5.predict(X_for_test)



