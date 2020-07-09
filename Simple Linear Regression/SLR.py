# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


pwd
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('original.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Diving data in two parts for Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

#importing Linear Regression Library
from sklearn.linear_model import LinearRegression

s_regression = LinearRegression()
s_regression.fit(X_train, y_train)
y_predicted = s_regression.predict(X_test)

#Plotting Training results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, s_regression.predict(X_train), color = 'blue')
plt.title('Study_Hours VS Exam_Score (training set)')
plt.xlabel('Study_Hours')
plt.ylabel('Exam_Score')
plt.show()

#Plotting Testing results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, s_regression.predict(X_test), color = 'red')
plt.title('Study_Hours VS Exam_Score (testing set)')
plt.xlabel('Study_hours')
plt.ylabel('Exam_Score')
plt.show()