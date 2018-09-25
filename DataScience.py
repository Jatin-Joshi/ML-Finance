
#Importing libraries and packages

import pandas as pd
import numpy as np
import scipy
from pandas import Series, DataFrame
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from collections import Counter

#Plotting Parameters

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

#Pulling Data

data = pd.read_csv('AAPL.csv', parse_dates = ['Date'], index_col=['Date'])
data = data[['Open', 'High', 'Low', 'Close']]

#One Day Lag

data['Open'] = data['Open'].shift(1)
data['High'] = data['High'].shift(1)
data['Low'] = data['Low'].shift(1)
data['Close'] = data['Close'].shift(1)

#Removing NaN

data = data.dropna()

#Removing Duplicates

data = data.drop_duplicates()

#Creating Exponential Moving Average

ma10 = data.ewm(span=10, min_periods = 0, adjust=False, ignore_na =False).mean()
ma15 = data.ewm(span=15, min_periods = 0, adjust=False, ignore_na =False).mean()
ma30 = data.ewm(span=30, min_periods = 0, adjust=False, ignore_na =False).mean()

#Using Closing Price and 50 Day EMA of Closing Price to create new dataframe


x = data[['Close']]
y = x.ewm(span=50, min_periods = 0, adjust=False, ignore_na =False).mean()
y.columns = ['EMA50']

test = pd.merge(x, y, left_index = True, right_index = True)

#Start of Linear Regression

#Plotting Relationships

correlation = sb.pairplot(test)

#Printing Out Correlation Matrix to see r value of Close vs EMA50 pair (Notice it is very strong, investigate other indicators)

print test.corr()

#Instantiate Linear Regression Object

LinReg = LinearRegression(normalize=True)

#Call a fit method off model and pass in our predictor variables, as well as our predictant

LinReg.fit(x, y)

#Model's multiple R square score

score = LinReg.score(x,y)

#Start of Logistic Regression'

z = x.ewm(span=15, min_periods = 0, adjust=False, ignore_na =False).mean()
z.columns = ['EMA15']
predictive = pd.merge(y, z, left_index = True, right_index = True)
Ldata = pd.merge(x, predictive, left_index = True, right_index = True)

#Checking for independence between features

independence = sb.regplot(x='EMA50', y = 'EMA15', data=Ldata, scatter = True)

#Checking for independence between predictive features (Want negative correlation, look for other indicators)

spearmanr_coefficient, p_value = spearmanr(y, z)
print 'Spearman Rank Correlation Coefficient %0.3f' % (spearmanr_coefficient)

#Deploying and Evaluating Model

X = scale(predictive)

LogReg = LogisticRegression()

LogReg.fit(X, x)
print LogReg.score(X,x)