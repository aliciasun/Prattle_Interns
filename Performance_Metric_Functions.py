
# coding: utf-8

# In[2]:

import dateutil.parser
import json
import matplotlib.pyplot as plt
import numpy 
import pandas as pd
import pandas.io.data
import requests
import datetime
import math
import operator
import scipy
import requests
import calendar
import statsmodels.api as sm


# In[3]:

#NEEDS COMMENTS FROM ALEX
def isFirstDay(date, dic1):
    d = date.to_datetime()
    for i in range (1,32):
        if dic1[d.month] == d.year:
            return False
        else:
            if d.day==i:
                dic1[d.month] = d.year
                return True
#Are you sure its not this?
#def isFirstDay(date):
    #return date.to_datetime().day == 1


# In[4]:

#returns a dictionary with the first day of each year
#input dic must be an empty dictionary
def isFirstDayYear(date, dic):
    d = date.to_datetime()
    for i in range (1,32):
        if d.year in dic:
            return False
        else:
            if d.day==i:
                dic[d.year] = True
                return True


# In[5]:

#Calculates the yearly returns from the price column in dataframe df. Dataframe df needs to have 'Date' column 
#along with 'Price' column. Creates a Firstdayyear column of booleans in df.
#yr_returns is a list with floats
def yearlyReturns(df):
    hasFirstDay = {}
    df['Firstdayyear'] = df.loc[:, 'Date'].apply(isFirstDayYear, dic=hasFirstDay)
    first_year_adj_close = []
    last_year_adj_close = []
    first_year = df[df['Firstdayyear']==True]
    last_year = df[df['Firstdayyear'] == True]
    first_year = first_year.drop(first_year.index[len(first_year)-1])
    last_year = last_year.drop(last_year.index[0])
    for i in first_year.index:
        first_year_adj_close.append(first_year['Price'][i])
    
    for i in last_year.index:
        last_year_adj_close.append(last_year['Price'][i])
    yr_returns = [(i - j)/i for i, j in zip(last_year_adj_close, first_year_adj_close)]
    return yr_returns


# In[6]:

#Calculates the monthly returns from the Price column in dataframe df
#Dataframe df must have Date column as well as Price column
#Adds Firstdaymonth column to df
#output is a list of floats
def monthlyReturns(df):
    hasFirstDayMonth = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    df['Firstdaymonth'] = df.loc[:, 'Date'].apply(isFirstDay, dic1=hasFirstDayMonth)
    first_month_adj_close = []
    last_month_adj_close = []
    first_month = df[df['Firstdaymonth']==True]
    last_month = df[df['Firstdaymonth']==True]
    last_month = last_month.drop(last_month.index[0])
    first_month = first_month.drop(first_month.index[len(first_month) - 1])
    for i in first_month.index:
        first_month_adj_close.append(first_month['Price'][i])
    
    for i in last_month.index:
        last_month_adj_close.append(last_month['Price'][i])
    monthly_returns = [(i - j)/i for i, j in zip(last_month_adj_close, first_month_adj_close)]
    return monthly_returns


# In[7]:

#adjust std depending on annualized range
#Takes in monthly or yearly returns (list of floats)
#outputs the sharpe_Ratio as a float
def sharpe_Ratio(returns):
    std = (returns.std()*(12**.5)) 
    return (returns.mean() - .02)/std


# In[8]:

#Takes in monthly or yearly returns (list of floats) and the DataFrame asset1Data
#asset1Data must have Date column along with Price column
#outputs the alpha as a float
def alpha(returns, asset1Data):
    portfolio_total_return = returns[len(returns)-1] - returns[0]
    asset1_monthly_return = monthlyReturns(asset1Data)
    asset1_total_return = asset1_monthly_return[len(asset1_monthly_return) - 1] - asset1_monthly_return[0]
    return portfolio_total_return - .02 - (1.0 * (asset1_total_return - .02))


# In[9]:

#returns list of maximum draw downs as floats 
#input must be dataframe with Price column
#difficult to interpret. At first I didn't think the code would work, but trust it. It works
def maxDrawDown(df):
    initial_max = 0
    initial_index = 0
    initial_index_integer = 0
    #initialize array of local maximums
    maximums = []
    initial_max = df["Price"][initial_index_integer]
    maximums.append(initial_max)
    #initialize array of the indexes of the local maximums
    indexes = []
    indexes.append(initial_index_integer)
    #initialize array of local minimums between maximums
    minimums = []
    #fill arrays with local maximums and their indexes
    for i in range(len(df)):
        if df["Price"][i] - df["Price"][initial_index_integer] > 0:
            initial_max = df["Price"][i]
            initial_index = df.index[i]
            initial_index_integer = df.index.get_loc(initial_index)
            maximums.append(initial_max)
            indexes.append(initial_index_integer)
    #fill array with the minimus between each 2 local maximums
    minimum = 1000000000
    for i in range(len(indexes)-1):
        for j in range(indexes[i], indexes[i + 1]):
            if df['Price'][j] < minimum:
                minimum = df['Price'][j]
        minimums.append(minimum)
        minimum = 1000000000
    #initialize max_draw_downs array
    max_draw_downs = []
    max_draw_down = 0
    #fill array with the max draw downs
    for i in range(len(maximums)-1): 
        max_draw_down = ((minimums[i] - maximums[i])/maximums[i])
        max_draw_downs.append(max_draw_down)
    return max_draw_downs


# In[10]:

#Takes in dataframe with Price and Date columns 
#returns the annualized compound return from the dataframe
def annualizedCompoundReturn(df):
    return (df.Price[len(df.Price)-1]/df.Price[0])**(1/float(len(yearlyReturns(df))))-1

