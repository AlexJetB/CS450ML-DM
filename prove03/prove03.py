# -*- coding: utf-8 -*-
"""
Created on Wed Oct 2 11:57:00 2019

@author: Alex J. Baker
"""
## START FROM CLASS INSTRUCTIONS ##
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
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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

        classes = np.unique(targets[indices[:k]])
        if len(classes)==1:
            closest[n] = np.unique(classes)
        else:
            counts = np.zeros(max(classes)+1)
            for i in range(k):
                counts[targets[indices[i]]] += 1
            closest[n] = np.max(counts)

    return closest

def separate_data_targets(data, targets, panda_arr):

    target_np = panda_arr[targets].to_numpy()
    if (len(targets) == 1):
        target_np = target_np.flatten()

    data_np = panda_arr[data].to_numpy()
    if (len(data) == 1):
        data_np = data_np.flatten()

    return data_np, target_np


data = iris.data
#print(data)
targets = iris.target

#print(y)
# Randomize data, 30% test, 70% train
data_train, data_test, targets_train, targets_test = train_test_split(data, targets, train_size=0.7,
                 test_size=0.3)

# Car data: Start
headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "condition"]
car_data = pd.read_csv("files/car.data",
                        header=None, names=headers, skipinitialspace=True,
                        na_values=["?"], index_col=False)

# Car data: Missing value handling
car_data[car_data.isna().any(axis=1)]
car_data.isna().any()

# Encoding

# Label
car_data["buying"] = car_data["buying"].astype("category")
car_data["buying_cat"] = car_data["buying"].cat.codes

# Label
car_data["maint"] = car_data["maint"].astype("category")
car_data["maint_cat"] = car_data["maint"].cat.codes

# Label
car_data["doors"] = car_data["doors"].astype("category")
car_data["doors_cat"] = car_data["doors"].cat.codes

# Label
car_data["persons"] = car_data["persons"].astype("category")
car_data["persons_cat"] = car_data["persons"].cat.codes

# label
car_data["lug_boot"] = car_data["lug_boot"].astype("category")
car_data["lug_boot_cat"] = car_data["lug_boot"].cat.codes

# label
car_data["safety"] = car_data["safety"].astype("category")
car_data["safety_cat"] = car_data["safety"].cat.codes

# label
car_data["condition"] = car_data["condition"].astype("category")
car_data["condition_cat"] = car_data["condition"].cat.codes

data_head = ["buying_cat", "maint_cat", "doors_cat", "persons_cat",
               "lug_boot_cat", "safety_cat"]
target_head = ["condition_cat"]

car_data, car_target = separate_data_targets(data_head, target_head, car_data)

car_data_train, car_data_test, car_target_train, car_target_test = train_test_split(car_data, car_target, train_size=0.7,
                 test_size=0.3)

# Using SKLearn kNN classifier with formula from prove02
kArr=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,50,100]
accuracyDict = {}
for k in kArr:
    kNNclassifier = KNeighborsClassifier(n_neighbors=k)
    kNNclassifier.fit(car_data_train, car_target_train)
    skresult = kNNclassifier.predict(car_data_test)

#    result = k_Nearest_Neighbor(k, car_data_train, car_target_train, car_data_test)

    accuracy = accuracy_score(car_target_test, skresult)
#    print("SKLearn CARS: Accuracy of: ", accuracy, " for k=", k)
    accuracy = accuracy_score(car_target_test, skresult)
#    print("Custom CARS: Accuracy of: ", accuracy, " for k=", k)
    accuracyDict[k] = accuracy

sortedDict = sorted(accuracyDict.items(), key=lambda x: x[1], reverse=True)
print("CAR Most accurate k:", sortedDict[0][0])
print("CAR With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))
# Noneeded

headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin"]
mpg_data_full = pd.read_csv("files/auto-mpg.data",
                        header=None, names=headers, skipinitialspace=True,
                        na_values=["?"], delim_whitespace=True, index_col=False)

# Drop only 6 rows of NA
mpg_data_full = mpg_data_full.dropna()

data_head = ["cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin"]
target_head = ["mpg"]

mpg_data, mpg_target = separate_data_targets(data_head, target_head, mpg_data_full)

mpg_data_train, mpg_data_test, mpg_target_train, mpg_target_test = train_test_split(mpg_data, mpg_target, train_size=0.7,
                 test_size=0.3)

accuracyDict = {}
for k in kArr:
    kNNclassifier = KNeighborsRegressor(n_neighbors=k)
    kNNclassifier.fit(mpg_data_train, mpg_target_train)
    skresult = kNNclassifier.predict(mpg_data_test)

#    result = k_Nearest_Neighbor(k, car_data_train, car_target_train, car_data_test)

    accuracy = r2_score(mpg_target_test, skresult)
    print("SKLearn CARS: Accuracy of: ", accuracy, " for k=", k)

    accuracyDict[k] = accuracy

sortedDict = sorted(accuracyDict.items(), key=lambda x: x[1], reverse=True)
print("MPG Most accurate k:", sortedDict[0][0])
print("MPG With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))

headers = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "gaurdian", "traveltime", "studytime", "failures",
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
        "G1","G2","G3"]

stdt_data_full = pd.read_csv("files/student-mat.csv",
                        header=None, names=headers, skipinitialspace=True,
                        na_values=["?"], sep=";", index_col=False)


