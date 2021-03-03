# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:07:40 2021

@author: Abhrajyoti Pal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

header_list = ["x1","x2","x3","y"]
df = pd.read_csv("data.csv", names = header_list)

features = ["x1","x2","x3"]
X = df[features]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = GaussianNB()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#plot_confusion_matrix(clf, X_test, y_test)
#plt.show()

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print("Precision: "+str(precision_score(y_test, y_pred, average='micro')))
print("Recall: "+str(recall_score(y_test, y_pred, average='micro')))
print("F1: "+str(f1_score(y_test, y_pred, average='micro')))

clf1 = MultinomialNB()
clf = clf1.fit(X_train,y_train)
y_pred1 = clf1.predict(X_test)

#plot_confusion_matrix(clf1, X_test, y_test)
#plt.show()

print(confusion_matrix(y_test, y_pred))

print("Accuracy: "+str(accuracy_score(y_test, y_pred1)))
print("Precision: "+str(precision_score(y_test, y_pred1, average='micro')))
print("Recall: "+str(recall_score(y_test, y_pred1, average='micro')))
print("F1: "+str(f1_score(y_test, y_pred1, average='micro')))