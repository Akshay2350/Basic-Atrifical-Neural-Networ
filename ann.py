# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:23:47 2018

@author: AKSHAY
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13 ].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

#print (X=='France')
#print (np.sum(X=='France'))
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part-2 Making of ANN

# Importing Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing  the ANN
classifier = Sequential()

#Adding the input layer and First hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding Second Hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100 )
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Predicting a single new observation
new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])))
new_pred = (new_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating, Improving and Tuning the ANN

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#If you want to use K-fold cross validation direct
from keras.models import Sequential
from keras.layers import Dense 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10 )
mean = accuracies.mean()    