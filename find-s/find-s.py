# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

class findS():
    def __init__(self, data, data_test):
        self.trainingData = data.iloc[:, data.columns != data.columns[-1]]
        print(self.trainingData)
        self.targetData = data.iloc[:,-1:]
        self.hypothesis = self.trainingData.iloc[0]
        self.testingData = data_test
        self.flag = 1
        
    def makeConsistent(self):
        for i in range(1, len(self.trainingData)):
            for j in range(len(self.trainingData.columns)):
                if(self.targetData.iloc[i][0] == 'Yes'):
                    if(self.trainingData.iloc[i][j] != self.hypothesis[j]):
                        self.hypothesis[j] = 'any'
        print(self.hypothesis)
    
    def test(self):
        self.flag = 1
        for i in range(0, len(self.testingData)):
            for j in range(len(self.testingData.columns)):
                if(self.hypothesis.iloc[j] == 'any'):
                    self.flag = 1
                    continue
                elif(self.hypothesis.iloc[j] == self.testingData.iloc[i][j]):
                    self.flag = 1
                    continue
                else:
                    self.flag = 0
                    break
            if(self.flag == 1):
                print('true')
            else:
                print('false')
                

data = pd.read_csv('find-s-train.csv')
data_test = pd.read_csv('find-s-test.csv')
model = findS(data, data_test)
model.makeConsistent()
model.test()