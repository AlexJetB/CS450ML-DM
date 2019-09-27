# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:10:09 2019

@author: Alex J. Baker
"""
### START FROM CLASS INSTRUCTIONS ###
###################
# PROVE 02 is an addition to Prove 01
# SEE
# https://tinyurl.com/yxz9jnvy (StackOverflow post)
# for txt file reading technique
# AND
# https://pythonprogramming.net/reading-csv-files-python-3/
# for CSV file reading technique
###################

import numpy as np
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#### For txt files ###
#with open("files/textFile.txt", "r") as txt:
#    txtArr = [line.split() for line in txt]
#
#print(txtArr)
#
#### For CSV files ###
#with open("files/csvFile.csv") as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    csvArr = []
#    for row in readCSV:
#        print(row)
#        csvArr.append(row)
##        print(row[0])
##        print(row[0],row[1],row[2],)
#print(csvArr)

class HardCodedClassifier:
    def fit(self, data_train=[], targets_train=[]):
        print('Fitting data!')
    def predict(self, test_data=[]):
        predicted_data = []
        for item in test_data:
            predicted_data.append(0)
        return predicted_data
iris = datasets.load_iris()

class KNN_Classifier():
    k=1
    def __init__(self, k=1):
        self.k = k
    def fit(self, data_train=[], targets_train=[]):
        print("hello")
    def predict(self, test_data=[]):
        predicted_data = []
        for item in test_data:
            predicted_data.append(0)
        return predicted_data
iris = datasets.load_iris()

### FOR PROVE 02 ###
def k_Nearest_Neighbor(k,data,targets,inputs):

    nInputs = np.shape(inputs)[0] # Number of inputs
    closest = np.zeros(nInputs) # List of closest inputs

    for n in range(nInputs):
        # Compute distances (euclidean)
        distances = np.sum((data-inputs[n,:])**2, axis=1) # From book....

        # Sort the nearest neighbors from closest to furthest
        indices = np.argsort(distances, axis=0)

        classes = np.unique(targets[indices[:k]]) # Enumerate our classes
        if len(classes)==1:
            closest[n] = np.unique(classes) # All of the same class
        else:
            counts = np.zeros(max(classes)+1)
            for i in range(k):
                counts[targets[indices[i]]] += 1
            closest[n] = np.max(counts) # Classify according to the most counted class

    return closest

# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)
### END FROM CLASS INSTRUCTIONS ###

data = iris.data
#print(data)
targets = iris.target

#print(y)
# Randomize data, 30% test, 70% train
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, train_size=0.7,
                 test_size=0.3)

#### Sklearn example ###
clf = GaussianNB()
clf.fit(data_train, targets_train)

targets_predicted = clf.predict(data_test)

# Measure accuracy of fit
accuracy = accuracy_score(targets_test, targets_predicted)
print(accuracy)

### 'Custom' Classifier ###
clf = HardCodedClassifier()
clf.fit(data_train, targets_train)

targets_predicted = clf.predict(data_test)

print(targets_predicted)
print(targets_test)

# Measure accuracy of fit
accuracy = accuracy_score(targets_test, targets_predicted)
print(accuracy)

### kNN alg from sklearn and prove02 instructions
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(data_train, targets_train)
predictions = classifier.predict(data_test)
print(predictions)

accuracy = accuracy_score(targets_test, predictions)
print(accuracy)

### my own kNN alg ###
k=3
result = k_Nearest_Neighbor(k, data_train, targets_train, data_test)

print(result)
accuracy = accuracy_score(targets_test, result)
print("Accuracy of: ", accuracy, " for k=", k)

k=5
result = k_Nearest_Neighbor(k, data_train, targets_train, data_test)

print(result)
accuracy = accuracy_score(targets_test, result)
print("Accuracy of: ", accuracy, " for k=", k)


k=20
result = k_Nearest_Neighbor(k, data_train, targets_train, data_test)

print(result)
accuracy = accuracy_score(targets_test, result)
print("Accuracy of: ", accuracy, " for k=", k)

k=30
result = k_Nearest_Neighbor(k, data_train, targets_train, data_test)

print(result)
accuracy = accuracy_score(targets_test, result)
print("Accuracy of: ", accuracy, " for k=", k)
