# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:09:03 2019

@author: baker
"""

'''
On your terminal run:
pip install alpha_vantage
This also uses the pandas dataframe, and matplotlib, commonly used python packages
pip install pandas
pip install matplotlib
For the develop version run:
pip install git+https://github.com/RomelTorres/alpha_vantage.git@develop
'''

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

# Your key here
key = '3ROVSK34ZW6E1Z6D'
# Chose your output format, or default to JSON (python dict)
ts = TimeSeries(key, output_format='pandas')
ti = TechIndicators(key)

# Get the data, returns a tuple
# aapl_data is a pandas dataframe, aapl_meta_data is a dict
aapl_data, aapl_meta_data = ts.get_daily(symbol='AMZN')
# aapl_sma is a dict, aapl_meta_sma also a dict
aapl_sma, aapl_meta_sma = ti.get_sma(symbol='AMZN')


# Visualization
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
aapl_data['4. close'].plot()
plt.tight_layout()
plt.grid()
plt.show()