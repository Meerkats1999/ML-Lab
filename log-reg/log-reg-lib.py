# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:33:23 2021

@author: Abhrajyoti Pal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

header_list = ["x1","x2","y"]
df = pd.read_csv('Student-University.csv', names = header_list)

X = df[["x1", "x2"]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#plot_confusion_matrix(clf, X_test, y_test)
#plt.show()

print("Precision: "+str(precision_score(y_test, y_pred, average='micro')))
print("Recall: "+str(recall_score(y_test, y_pred, average='micro')))
print("F1: "+str(f1_score(y_test, y_pred, average='micro')))