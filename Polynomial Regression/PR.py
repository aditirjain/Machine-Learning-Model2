# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:23:21 2020

@author: Aditi Jain
"""
pwd

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv('reward_system.csv')

X = dataset.iloc[:,0:1].values
y = dataset.iloc[:, 1].values


from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color = 'black')
plt.title('Reward System (Linear Regression)')
plt.xlabel('Hours')
plt.ylabel('Points')
plt.show()


#Importing libraries
from sklearn.preprocessing import PolynomialFeatures

#Fitting Polynomial Regression to training data
polynomial_reg = PolynomialFeatures(degree=2)
X_polynomial = polynomial_reg.fit_transform(X)
linear_polynomial = LinearRegression()
linear_polynomial.fit(X_polynomial, y)

#Plotting Polynomial Regression
plt.scatter(X, y, color = 'black')
plt.plot(X, linear_polynomial.predict(polynomial_reg.fit_transform(X)), color = 'red')
plt.title('Reward System(Polynomial Regression)')
plt.Xlabel('Hours')
plt.ylabel('Points')
plt.show()

#predicting y value using Linear Regression
y_pred_by_simple_linear = linear_reg.predict([[90]])

#Predicting y value using Polynomial Regression
y_pred_by_polynomial = linear_polynomial.predict(polynomial_reg.fit_transform([[90]]))


#If we change degree to 3
polynomial_reg = PolynomialFeatures(degree=3)
X_polynomial = polynomial_reg.fit_transform(X)
linear_polynomial = LinearRegression()
linear_polynomial.fit(X_polynomial, y)


#Plotting Polynomial Regression
plt.scatter(X, y, color = 'black')
plt.plot(X, linear_polynomial.predict(polynomial_reg.fit_transform(X)), color = 'red')
plt.title('Reward System(Polynomial Regression)')
plt.Xlabel('Hours')
plt.ylabel('Points')
plt.show()

