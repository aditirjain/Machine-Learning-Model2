# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:06:56 2020

@author: Aditi Jain
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:37:13 2020

@author: Aditi Jain
"""

import pandas as pd 

dataset = pd.read_excel('alarm_files.xlsx')


X = dataset.iloc[:, [1, 2, 3, 4, 5] ].values     #input variables
y = dataset.iloc[:, -1].values        #target variable

#Encoding categorical data to create the dummy variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()


X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 4] = labelencoder.fit_transform(X[:, 4])

df1 = pd.DataFrame(X)


hotencoder = OneHotEncoder(categorical_features=[0])
X = hotencoder.fit_transform(X).toarray()

hotencoder = OneHotEncoder(categorical_features=[3])
X = hotencoder.fit_transform(X).toarray()

hotencoder = OneHotEncoder(categorical_features=[6])
X = hotencoder.fit_transform(X).toarray()

hotencoder = OneHotEncoder(categorical_features=[11])
X = hotencoder.fit_transform(X).toarray()

hotencoder = OneHotEncoder(categorical_features=[13])
X = hotencoder.fit_transform(X).toarray()



#Sampling the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#importing StandardScaler from sklearn.preprocessing to scale matrix of features
from sklearn.preprocessing import StandardScaler
scaled_X = StandardScaler()
X_train = scaled_X.fit_transform(X_train)
X_test = scaled_X.transform(X_test)

#importing logistic regression from sklearn.Linear_model to build logisticRegression classifier
from sklearn.linear_model import LogisticRegression


#fitting Logisyic Regression to training test
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predicting y values using predict method
y_pred = classifier.predict(X_test)

#Creating confusion matrix to find model prediction power
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


