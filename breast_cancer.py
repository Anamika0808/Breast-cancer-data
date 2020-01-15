# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:27:05 2019

@author: 91947
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors

df = pd.read_csv('data.csv')
df.replace('?', -99999, inplace = True)
df.drop(['id','Unnamed: 32'], axis = 1, inplace = True)

X =np.array(df.drop('diagnosis', axis=1)).astype(float)
X = preprocessing.scale(X)

y= np.array(df['diagnosis'].map({'M':1, 'B': 0}))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,  test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
pred = clf.predict(X_test)