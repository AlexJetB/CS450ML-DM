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
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
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

# One hot
buyDummies = pd.get_dummies(car_data["buying"], prefix = "buying")
car_data = pd.concat([car_data, buyDummies], axis=1)
#%%
# Label
car_data["maint"] = car_data["maint"].astype("category")
car_data["maint_cat"] = car_data["maint"].cat.codes

# One hot
maintDummies = pd.get_dummies(car_data["maint"], prefix = "maint")
car_data = pd.concat([car_data, maintDummies], axis=1)
#%%
# Label
car_data["doors"] = car_data["doors"].astype("category")
car_data["doors_cat"] = car_data["doors"].cat.codes

# One hot
doorsDummies = pd.get_dummies(car_data["doors"], prefix = "doors")
car_data = pd.concat([car_data, doorsDummies], axis=1)
#%%
# Label
car_data["persons"] = car_data["persons"].astype("category")
car_data["persons_cat"] = car_data["persons"].cat.codes

# One hot
personsDummies = pd.get_dummies(car_data["persons"], prefix = "persons")
car_data = pd.concat([car_data, personsDummies], axis=1)
#%%
# label
car_data["lug_boot"] = car_data["lug_boot"].astype("category")
car_data["lug_boot_cat"] = car_data["lug_boot"].cat.codes

# One hot
lugDummies = pd.get_dummies(car_data["lug_boot"], prefix = "lug_boot")
car_data = pd.concat([car_data, lugDummies], axis=1)
#%%
# label
car_data["safety"] = car_data["safety"].astype("category")
car_data["safety_cat"] = car_data["safety"].cat.codes

# One hot
safetyDummies = pd.get_dummies(car_data["safety"], prefix = "safety")
car_data = pd.concat([car_data, safetyDummies], axis=1)
#%%
# label
car_data["condition"] = car_data["condition"].astype("category")
car_data["condition_cat"] = car_data["condition"].cat.codes
#%%
data_head = ["buying_cat", "maint_cat", "doors_cat", "persons_cat",
               "lug_boot_cat", "safety_cat"]
data_head_OH = ["buying_high", "buying_low", "buying_med", "buying_vhigh",
                "maint_high", "maint_low", "maint_med", "maint_vhigh",
                "doors_2",  "doors_3",  "doors_4",  "doors_5more",
                "persons_2",  "persons_4",  "persons_more",
                "lug_boot_big",  "lug_boot_med",  "lug_boot_small",
                "safety_high", "safety_low", "safety_med"]

target_head = ["condition_cat"]
#%%
car_data_cat, car_target = separate_data_targets(data_head, target_head, car_data)

car_data_OH, car_target = separate_data_targets(data_head_OH, target_head, car_data)

car_data_train, car_data_test, car_target_train, car_target_test = train_test_split(car_data_cat, car_target, train_size=0.7,
                 test_size=0.3)
car_data_train_OH, car_data_test_OH, car_target_train, car_target_test = train_test_split(car_data_OH, car_target, train_size=0.7,
                 test_size=0.3)
#%%
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
print("CAR CAT ENC Most accurate k:", sortedDict[0][0])
print("CAR CAT ENC With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))
# Noneeded
#%%
for k in kArr:
    kNNclassifier = KNeighborsClassifier(n_neighbors=k)
    kNNclassifier.fit(car_data_train_OH, car_target_train)
    skresult = kNNclassifier.predict(car_data_test_OH)

#    result = k_Nearest_Neighbor(k, car_data_train, car_target_train, car_data_test)

    accuracy = accuracy_score(car_target_test, skresult)
#    print("SKLearn CARS: Accuracy of: ", accuracy, " for k=", k)
    accuracy = accuracy_score(car_target_test, skresult)
#    print("Custom CARS: Accuracy of: ", accuracy, " for k=", k)
    accuracyDict[k] = accuracy

sortedDict = sorted(accuracyDict.items(), key=lambda x: x[1], reverse=True)
print("CAR ONE HOT Most accurate k:", sortedDict[0][0])
print("CAR ONE HOT With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))

#%%
headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin"]
mpg_data_full = pd.read_csv("files/auto-mpg.data",
                        header=None, names=headers, skipinitialspace=True,
                        na_values=["?"], delim_whitespace=True, index_col=False)

# Drop only 6 rows of NA
mpg_data_full = mpg_data_full.dropna()

data_head = ["cylinders", "displacement", "horsepower", "weight"]
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
#    print("SKLearn MPG: Accuracy of: ", accuracy, " for k=", k)

    accuracyDict[k] = accuracy

sortedDict = sorted(accuracyDict.items(), key=lambda x: x[1], reverse=True)
print("MPG Most accurate k:", sortedDict[0][0])
print("MPG With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))

ax = plt.gca()
mpg_data_full.plot(kind='scatter', x='mpg', y='displacement', color='red', ax=ax)
#mpg_data_full.plot(kind='scatter', x='mpg', y='acceleration', color='blue', ax=ax)
mpg_data_full.plot(kind='scatter', x='mpg', y='weight', color='green', ax=ax)
mpg_data_full.plot(kind='scatter', x='mpg', y='horsepower', color='purple', ax=ax)
mpg_data_full.plot(kind='scatter', x='mpg', y='cylinders', color='orange', ax=ax)
#mpg_data_full.plot(kind='scatter', x='mpg', y='model_year', color='yellow', ax=ax)
#mpg_data_full.plot(kind='scatter', x='mpg', y='origin', color='teal', ax=ax)

plt.show()

# Drop origin, acceleration, and model_year
mpg_data_norm = mpg_data_full.drop(["origin","acceleration","model_year"], axis=1)

mpg_vals = mpg_data_norm.values
normalizer = preprocessing.Normalizer()
mpg_vals_scaled = normalizer.fit_transform(mpg_vals)
mpg_data_norm = pd.DataFrame(mpg_vals_scaled)

ax2 = plt.gca()
mpg_data_norm.plot(kind='scatter', x=4, y=0, color='orange', ax=ax2)
mpg_data_norm.plot(kind='scatter', x=4, y=1, color='red', ax=ax2)
mpg_data_norm.plot(kind='scatter', x=4, y=2, color='purple', ax=ax2)
mpg_data_norm.plot(kind='scatter', x=4, y=3, color='green', ax=ax2)

plt.show()

headers = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
        "G1","G2","G3"]

stdt_data_full = pd.read_csv("files/student-mat.csv",
                        header=0, skipinitialspace=True,
                        na_values=["?"], sep=";", index_col=False)

# No stated NA values, no need to remove #

# School is binary
stdt_data_full["school_GP"] = stdt_data_full.school.map({"GP": 1, "MS": 0})

# Sex is binary
stdt_data_full["isMale"] = stdt_data_full.sex.map({"M": 1, "F": 0})

# Age is numeric

# Address is binary
stdt_data_full["address_U"] = stdt_data_full.address.map({"U": 1, "R": 0})

# Famsize is binary
stdt_data_full["famsize_GT3"] = stdt_data_full.famsize.map({"GT3": 1, "LE3": 0})

# Pstatus is binary
stdt_data_full["Pstatus_T"] = stdt_data_full.Pstatus.map({"T": 1, "A": 0})

# Medu is numeric

# Fedu is numeric

# Mjob is nominal, label encode
stdt_data_full["Mjob"] = stdt_data_full["Mjob"].astype("category")
stdt_data_full["Mjob_cat"] = stdt_data_full["Mjob"].cat.codes

# Fjob is nominal, label encode
stdt_data_full["Fjob"] = stdt_data_full["Fjob"].astype("category")
stdt_data_full["Fjob_cat"] = stdt_data_full["Fjob"].cat.codes

# Reason is nominal, one hot encode
reasonDummies = pd.get_dummies(stdt_data_full["reason"], prefix = "reason")
stdt_data_full = pd.concat([stdt_data_full, reasonDummies], axis=1)

# Gaurdian is nominal, one hot encode
guardianDummies = pd.get_dummies(stdt_data_full["guardian"], prefix = "guardian")
stdt_data_full = pd.concat([stdt_data_full, guardianDummies], axis=1)

# Traveltime is numeric

# Studytime is numeric

# Failures is numeric

# Schoolsup is binary
stdt_data_full["schoolsup_Yes"] = stdt_data_full.schoolsup.map({"yes": 1, "no": 0})

# Famsup is binary
stdt_data_full["famsup_Yes"] = stdt_data_full.famsup.map({"yes": 1, "no": 0})

# Paid is binary
stdt_data_full["paid_Yes"] = stdt_data_full.paid.map({"yes": 1, "no": 0})

# Activities is binary
stdt_data_full["activities_Yes"] = stdt_data_full.activities.map({"yes": 1, "no": 0})

# Nursery is binary
stdt_data_full["nursery_Yes"] = stdt_data_full.nursery.map({"yes": 1, "no": 0})

# Higher is binary
stdt_data_full["higher_Yes"] = stdt_data_full.higher.map({"yes": 1, "no": 0})

# Internet is binary
stdt_data_full["internet_Yes"] = stdt_data_full.internet.map({"yes": 1, "no": 0})

# Romantic is binary
stdt_data_full["romantic_Yes"] = stdt_data_full.romantic.map({"yes": 1, "no": 0})

# Famrel is numeric

# Freetime is numeric

# Goout is numeric

# Dalc is numeric

# Walc is numeric

# Health is numeric

# Absences is numeric

# G1, G2, G3 is numeric

data_head = ["school_GP", "isMale", "age", "address_U", "famsize_GT3",
             "Pstatus_T", "Medu", "Fedu", "Mjob_cat", "Fjob_cat", "reason_course",
             "reason_home", "reason_other", "reason_reputation",
             "guardian_father", "guardian_mother", "guardian_other",
             "traveltime", "studytime", "failures", "schoolsup_Yes",
             "famsup_Yes","paid_Yes","activities_Yes","nursery_Yes",
             "higher_Yes","internet_Yes","romantic_Yes","famrel",
             "freetime","goout","Dalc","Walc","health","absences","G1","G2"]

target_head = ["G3"]

stdt_data, stdt_target = separate_data_targets(data_head, target_head, stdt_data_full)

stdt_data_train, stdt_data_test, stdt_target_train, stdt_target_test = train_test_split(stdt_data, stdt_target, train_size=0.7,
                 test_size=0.3)

accuracyDict = {}
for k in kArr:
    kNNclassifier = KNeighborsRegressor(n_neighbors=k)
    kNNclassifier.fit(stdt_data_train, stdt_target_train)
    skresult = kNNclassifier.predict(stdt_data_test)

#    result = k_Nearest_Neighbor(k, car_data_train, car_target_train, car_data_test)

    accuracy = r2_score(stdt_target_test, skresult)
#    print("SKLearn MPG: Accuracy of: ", accuracy, " for k=", k)

    accuracyDict[k] = accuracy

sortedDict = sorted(accuracyDict.items(), key=lambda x: x[1], reverse=True)
print("STDT Most accurate k:", sortedDict[0][0])
print("STDT With accuracy of: ", "{:.2%}".format(sortedDict[0][1]))
