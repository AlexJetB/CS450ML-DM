# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:23:02 2019


@author: baker
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# START Iris
iris = datasets.load_iris()

data_i = iris.data

target_i = iris.target

data_i_train, data_i_test, targets_i_train, targets_i_test = train_test_split(data_i, target_i, train_size=0.7,
                 test_size=0.3)

# No depth testing
# Default (gini, best) decision tree
dtc1 = tree.DecisionTreeClassifier()

dtc1 = dtc1.fit(data_i_train, targets_i_train)

#tree.plot_tree(dtc1)

predictions_i = dtc1.predict(data_i_test)

accuracy = accuracy_score(targets_i_test, predictions_i)
print(accuracy)

# Entropy Best split decision tree.
dtc2 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_split=4,
                                   min_samples_leaf=3)

dtc2 = dtc2.fit(data_i_train, targets_i_train)

tree.plot_tree(dtc2)

predictions_i = dtc2.predict(data_i_test)

accuracy = accuracy_score(targets_i_test, predictions_i)
print(accuracy)