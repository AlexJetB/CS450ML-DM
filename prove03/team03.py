# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:11:24 2019

@author: baker
"""

import pandas as pd

headers = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "sex",
           "capital_gain", "capital_loss", "hours_per_week", "native_country"]

cns_data = pd.read_csv("files/adult.data", header=None, skipinitialspace=True,
                          names=headers, na_values=["?"], index_col=False)

print(cns_data)
print(cns_data.columns)
print(cns_data.dtypes)

print(cns_data.age.median())
print(cns_data.native_country.value_counts())
print(cns_data.dtypes)

### Drop rows (optional) ###
#beforeDrop = cns_data.shape[0]
#print('Number of Rows before NA drop: ', beforeDrop)
#
#cns_data = cns_data.dropna()
#
#postDrop = cns_data.shape[0]
#print('Number of Rows after NA drop: ', postDrop)

### bro. Burton's unknown solution to NaN ###
cns_data[cns_data.isna().any(axis=1)]
cns_data.isna().any()

cns_data.workclass = cns_data.workclass.fillna("unknown")
cns_data.native_country = cns_data.native_country.fillna("unknown")
cns_data.occupation = cns_data.occupation.fillna("unknown")

cns_data[cns_data.isna().any(axis=1)]
cns_data.isna().any()

cns_data["workclass"] = cns_data["workclass"].astype('category')
cns_data["workclass_cat"] = cns_data["workclass"].cat.codes
cns_data.head()

cns_data["education"] = cns_data["education"].astype('category')
cns_data["education_cat"] = cns_data["education"].cat.codes
cns_data.head()

cns_data["marital_status"] = cns_data["marital_status"].astype('category')
cns_data["marital_status_cat"] = cns_data["marital_status"].cat.codes
cns_data.head()

cns_data["occupation"] = cns_data["occupation"].astype('category')
cns_data["occupation_cat"] = cns_data["occupation"].cat.codes
cns_data.head()

cns_data["relationship"] = cns_data["relationship"].astype('category')
cns_data["relationship_cat"] = cns_data["relationship"].cat.codes
cns_data.head()

## One hot encoding makes more sense here ##
cns_data["race"] = cns_data["race"].astype('category')
cns_data["race_cat"] = cns_data["race"].cat.codes
cns_data.head()

cns_data["sex"] = cns_data["sex"].astype('category')
cns_data["sex_cat"] = cns_data["sex"].cat.codes
cns_data.head()

cns_data["native_country"] = cns_data["native_country"].astype('category')
cns_data["native_country_cat"] = cns_data["native_country"].cat.codes
cns_data.head()

print(cns_data)
print(cns_data.dtypes)
#print(cns_data)