import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import svm 
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("heart.csv")

x = data.drop('target',axis = 1) 
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# Model Accuracy: how often is the classifier correct?
#print(ml.score(x_test,y_test))
cm = confusion_matrix(y_test,y_pred)
print(cm)
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
print('False Positives\n {}'.format(FP))
print('False Negetives\n {}'.format(FN))
print('True Positives\n {}'.format(TP))
print('True Negetives\n {}'.format(TN))
TPR = TP/(TP+FN)
print('Sensitivity \n {}'.format(TPR))
TNR = TN/(TN+FP)
print('Specificity \n {}'.format(TNR))
Precision = TP/(TP+FP)
print('Precision \n {}'.format(Precision))
Recall = TP/(TP+FN)
print('Recall \n {}'.format(Recall))
Acc = (TP+TN)/(TP+TN+FP+FN)
print('Áccuracy \n{}'.format(Acc))
Fscore = 2*(Precision*Recall)/(Precision+Recall)
print('FScore \n{}'.format(Fscore))

#plot_confusion_matrix(clf, X_test, y_test)
#plt.show()

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print("Precision: "+str(precision_score(y_test, y_pred, average='micro')))
print("Recall: "+str(recall_score(y_test, y_pred, average='micro')))
print("F1: "+str(f1_score(y_test, y_pred, average='micro')))

