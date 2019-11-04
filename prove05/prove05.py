# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:23:02 2019

Default decision tree attributes sklearn
criterion="gini", splitter="best",
                                    max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0,
                                    max_features=None,
                                    random_state=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    class_weight=None,
                                    presort=True

@author: baker
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn import tree
from IPython.display import Image
import pydotplus

def separate_data_targets(data, targets, panda_arr):

    target_np = panda_arr[targets].to_numpy()
    if (len(targets) == 1):
        target_np = target_np.flatten()

    data_np = panda_arr[data].to_numpy()
    if (len(data) == 1):
        data_np = data_np.flatten()

    return data_np, target_np

# START Iris
iris = datasets.load_iris()

data_i = iris.data

target_i = iris.target

data_i_train, data_i_test, targets_i_train, targets_i_test = train_test_split(data_i, target_i, train_size=0.7,
                 test_size=0.3)

# Entropy Best split decision tree.
dtc_i = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)

dtc_i = dtc_i.fit(data_i_train, targets_i_train)

# Consider different decision tree
#tree.plot_tree(dtc_i)

dot_data_i = tree.export_graphviz(dtc_i, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)

graph = pydotplus.graph_from_dot_data(dot_data_i)

Image(graph.create_png())

# Create PNG
graph.write_png("iris.png")

predictions_i = dtc_i.predict(data_i_test)

accuracy = accuracy_score(targets_i_test, predictions_i)
print("Iris accuracy:",accuracy)



# START LENSES

headers_l = [#"age_of_patient",
        "patient_num", "age_of_patient", "spectacle_prescription", "astigmatic", "tear_production_rate", "fitting"]
data_l = pd.read_csv('files/lenses.data',
                     header=None, names=headers_l, skipinitialspace=True,
                     sep=" ", na_values=["?"], index_col=False)
# Not needed...
#data_l = data_l.drop(['patient_num'], axis=1)
#data_l = data_l.drop(['age_of_patient'], axis=1)

# One Hot encoding
# age
ageDummies = pd.get_dummies(data_l["age_of_patient"], prefix = "age")
data_l = pd.concat([data_l, ageDummies], axis=1)

# prescription
prescDummies = pd.get_dummies(data_l["spectacle_prescription"], prefix = "presc")
data_l = pd.concat([data_l, prescDummies], axis=1)

# astigmatic
astigDummies = pd.get_dummies(data_l["astigmatic"], prefix = "astigmatic")
data_l = pd.concat([data_l, astigDummies], axis=1)

# tear rate
tearDummies = pd.get_dummies(data_l["tear_production_rate"], prefix = "tear")
data_l = pd.concat([data_l, tearDummies], axis=1)

# Fitting, the target
fitDummies = pd.get_dummies(data_l["fitting"], prefix = "fitting")
data_l = pd.concat([data_l, fitDummies], axis=1)


# One hot encoding
data_names_l_OH = [#"age_1", "age_2", "age_3",
                   "presc_1", "presc_2",
                   "astigmatic_1", "astigmatic_2",
                   "tear_1", "tear_2"]

# Label encoding
data_names_l = ["spectacle_prescription", "astigmatic",
                "tear_production_rate"]

target_names_l = ["fitting"]

target_names_l_OH = ["fitting_1",
                     "fitting_2",
                     "fitting_3"]

# Label
data_l_label, target_l = separate_data_targets(data_names_l,target_names_l,data_l)
data_l_train, data_l_test, targets_l_train, targets_l_test = train_test_split(data_l_label, target_l, train_size=0.7,
                 test_size=0.3)

# One Hot
data_l_OH, target_l_OH = separate_data_targets(data_names_l_OH,target_names_l_OH,data_l)
data_l_OH_train, data_l_OH_test, targets_l_OH_train, targets_l_OH_test = train_test_split(data_l_OH, target_l_OH, train_size=0.7,
                 test_size=0.3)


dtc_l = tree.DecisionTreeClassifier(criterion="entropy", splitter="random",
                                    max_depth=3, min_samples_split=2,
                                    min_samples_leaf=2,
                                    min_weight_fraction_leaf=0,
                                    max_features=None,
                                    random_state=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    class_weight=None,
                                    presort=False)

dtc_l_OH = tree.DecisionTreeClassifier(criterion="entropy", splitter="random",
                                    max_depth=4, min_samples_split=2,
                                    min_samples_leaf=2,
                                    min_weight_fraction_leaf=0,
                                    max_features=None,
                                    random_state=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    class_weight="balanced",
                                    presort=False)

dtc_l = dtc_l.fit(data_l_train, targets_l_train)

dot_data_l = tree.export_graphviz(dtc_l, out_file=None,
                                feature_names=data_names_l,
                                class_names=["hard_contact_lenses",
                                             "soft_contact_lenses",
                                             "no contact lenses"])

graph = pydotplus.graph_from_dot_data(dot_data_l)

Image(graph.create_png())

# Create PNG
graph.write_png("lenses.png")

# Predict
predictions_l = dtc_l.predict(data_l_test)

accuracy = accuracy_score(targets_l_test, predictions_l)
print("Lense, label encoded accuracy:", accuracy)


dtc_l_OH = dtc_l_OH.fit(data_l_OH_train, targets_l_OH_train)

dot_data_l_OH = tree.export_graphviz(dtc_l_OH, out_file=None,
                                feature_names=data_names_l_OH,
                                class_names=["hard_contact_lenses",
                                             "soft_contact_lenses",
                                             "no contact lenses"])

graph = pydotplus.graph_from_dot_data(dot_data_l_OH)

Image(graph.create_png())

# Create PNG
graph.write_png("lenses_OH.png")

# Predict
predictions_l_OH = dtc_l_OH.predict(data_l_OH_test)

accuracy_OH = accuracy_score(targets_l_OH_test, predictions_l_OH)
print("Lens, one hot accuracy:", accuracy_OH)

#%%
# START 1984 HOUSE VOTES
headers_v = ["party",
             "issue1",
             "issue2",
             "issue3",
             "issue4",
             "issue5",
             "issue6",
             "issue7",
             "issue8",
             "issue9",
             "issue10",
             "issue11",
             "issue12",
             "issue13",
             "issue14",
             "issue15",
             "issue16"]

data_v = pd.read_csv('files/house-votes-84.data',
                     header=None, names=headers_v, skipinitialspace=True,
                     sep=",", index_col=False)

# Consider "present" votes as nay
data_v = data_v.replace("?", "n")
#data_v = data_v.replace("?", 3) # "present"/absent votes as 3rd option.

# Consider nay("n") = 0, yea("y") = 1
data_v = data_v.replace("n", 0)
data_v = data_v.replace("y", 1)

# Consider Republican = 1, Democrat = 2
data_v = data_v.replace("republican", 1)
data_v = data_v.replace("democrat", 2)

data_names_v = ["issue1",
             "issue2",
             "issue3",
             "issue4",
             "issue5",
             "issue6",
             "issue7",
             "issue8",
             "issue9",
             "issue10",
             "issue11",
             "issue12",
             "issue13",
             "issue14",
             "issue15",
             "issue16"]

target_names_v = ["party"]

data_v, target_v = separate_data_targets(data_names_v,target_names_v,data_v)

data_v_train, data_v_test, targets_v_train, targets_v_test = train_test_split(data_v, target_v, train_size=0.7,
                 test_size=0.3)

dtc_v = tree.DecisionTreeClassifier(criterion="entropy", splitter="best",
                                    max_depth=None, min_samples_split=70,
                                    min_samples_leaf=70,
                                    min_weight_fraction_leaf=0,
                                    max_features=16,
                                    random_state=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    class_weight=None,
                                    presort=False)

dtc_v = dtc_v.fit(data_v_train, targets_v_train)

dot_data_v = tree.export_graphviz(dtc_v, out_file=None,
                                feature_names=data_names_v,
                                class_names=["Republican",
                                             "Democrat"])

graph = pydotplus.graph_from_dot_data(dot_data_v)

Image(graph.create_png())

# Create PNG
graph.write_png("voting.png")

# Predict
predictions_v = dtc_v.predict(data_v_test)

accuracy = accuracy_score(targets_v_test, predictions_v)
print("Voting prediction accuracy:", accuracy)

#%%

# START Student data from prove03

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
stdt_data_full["age"] = pd.cut(stdt_data_full.age, 2).cat.codes

# Address is binary
stdt_data_full["address_U"] = stdt_data_full.address.map({"U": 1, "R": 0})

# Famsize is binary
stdt_data_full["famsize_GT3"] = stdt_data_full.famsize.map({"GT3": 1, "LE3": 0})

# Pstatus is binary
stdt_data_full["Pstatus_T"] = stdt_data_full.Pstatus.map({"T": 1, "A": 0})

# Medu is numeric, cat

# Fedu is numeric, cat

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

# Traveltime is numeric, cat

# Studytime is numeric, cat

# Failures is numeric, cat

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
stdt_data_full["famrel"] = pd.cut(stdt_data_full.famrel, 2).cat.codes

# Freetime is numeric
stdt_data_full["freetime"] = pd.cut(stdt_data_full.freetime, 2).cat.codes

# Goout is numeric
stdt_data_full["goout"] = pd.cut(stdt_data_full.goout, 2).cat.codes

# Dalc is numeric
stdt_data_full["Dalc"] = pd.cut(stdt_data_full.Dalc, 2).cat.codes

# Walc is numeric
stdt_data_full["Walc"] = pd.cut(stdt_data_full.Walc, 2).cat.codes

# Health is numeric
stdt_data_full["health"] = pd.cut(stdt_data_full.health, 2).cat.codes

# Absences is numeric, no cat at all
stdt_data_full["absences"] = pd.cut(stdt_data_full.absences, 4).cat.codes

# G1, G2, G3 is numeric
stdt_data_full["G1"] = pd.cut(stdt_data_full.G1, 4).cat.codes
stdt_data_full["G2"] = pd.cut(stdt_data_full.G2, 4).cat.codes
stdt_data_full["G3"] = pd.cut(stdt_data_full.G3, 4).cat.codes

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

dtc_s = tree.DecisionTreeClassifier(criterion="entropy", splitter="best",
                                    max_depth=None, min_samples_split=2,
                                    min_samples_leaf=100,
                                    min_weight_fraction_leaf=0,
                                    max_features=None,
                                    random_state=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,
                                    class_weight=None,
                                    presort=False)

dtc_s = dtc_s.fit(stdt_data_train, stdt_target_train)

dot_data_s = tree.export_graphviz(dtc_s, out_file=None,
                                feature_names=data_head,
                                class_names=["G3: 0 < x <= 5",
                                             "G3: 5 < x <= 10",
                                             "G3: 10 < x <= 15",
                                             "G3: 15 < x <= 20"])

graph = pydotplus.graph_from_dot_data(dot_data_s)

Image(graph.create_png())

# Create PNG
graph.write_png("students.png")

# Predict
predictions_s = dtc_s.predict(stdt_data_test)

accuracy = accuracy_score(stdt_target_test, predictions_s)
print("GPA prediction accuracy:", accuracy)

