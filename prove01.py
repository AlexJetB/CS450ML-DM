# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:10:09 2019

@author: Alex J. Baker
"""
### START FROM CLASS INSTRUCTIONS ###
################### 
# SEE 
# https://tinyurl.com/yxz9jnvy (StackOverflow post)
# for txt file reading technique
# AND 
# https://pythonprogramming.net/reading-csv-files-python-3/
# for CSV file reading technique
###################

#import numpy as np #likely not needed
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### For txt files ###
with open("files/textFile.txt", "r") as txt:
    txtArr = [line.split() for line in txt]

print(txtArr)

### For CSV files ###
with open("files/csvFile.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    csvArr = []
    for row in readCSV:
        print(row)
        csvArr.append(row)
#        print(row[0])
#        print(row[0],row[1],row[2],)      
print(csvArr)

class HardCodedClassifier:
    def fit(self, data_train=[], targets_train=[]):
        print('Fitting data!')
    def predict(self, test_data=[]):
        predicted_data = []
        for item in test_data:
            predicted_data.append(0)
        return predicted_data
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)
### END FROM CLASS INSTRUCTIONS ###

data = iris.data
targets = iris.target
print(len(targets))

#print(y) 
# Randomize data, 30% test, 70% train
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, train_size=0.7, 
                 test_size=0.3)

#### Sklearn example ###
#clf = GaussianNB()
#clf.fit(data_train, targets_train)
#
#targets_predicted = clf.predict(data_test)
#
#print(targets_predicted)
#print(targets_test)
#
## Measure accuracy of fit
#accuracy = accuracy_score(targets_test, targets_predicted)
#print(accuracy)

### 'Custom' Classifier ###
clf = HardCodedClassifier()
clf.fit(data_train, targets_train)

targets_predicted = clf.predict(data_test)

print(targets_predicted)
print(targets_test)

# Measure accuracy of fit
accuracy = accuracy_score(targets_test, targets_predicted)
print(accuracy)