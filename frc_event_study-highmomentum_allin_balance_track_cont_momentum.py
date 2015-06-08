
# coding: utf-8

# In[1]:

# This version tracks the portfolio balance on a day to day basis no matter trading happens or not. 
# strategy is all in and all out
get_ipython().magic(u'matplotlib inline')
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
# Config
root_url = 'https://pa-api-dev.net/api'
api_key = 'uzuhm52a647vSn5wo3XMI4ZEDJ6ei0'
headers={'Authorization': 'Bearer {0}'.format(api_key)}
def parseDate(date):
    try:
        return dateutil.parser.parse(date).date()
    # parse the date to timestamp such that it can be resampled. Datetime objects cannot be resampled.
    # parse the json date to the pandas datetime
    except:
        return None

def getBankData(bank):
    response = requests.get(url="{0}/documents/bank/{1}".format(root_url, bank),
             headers=headers, verify=False)
    json_data = json.loads(response.content)
    df = pd.DataFrame(json_data)
    
    #only want the date_scored and score column
    df = df.loc[:,['date_scored', 'score']]
    
    df['date_scored'] = df['date_scored'].apply(parseDate)
    
    return df


# In[2]:

frcData = getBankData('frc')
# remove all the null values from the dataframe
frcData = frcData[~(frcData['score'].isnull() | frcData['date_scored'].isnull())]


# In[4]:

# getting the mean frc score on a date with several releases
mean_frc = frcData.groupby('date_scored')['score'].mean()
md=pd.DataFrame(mean_frc)

# make a copy of the dataframe for further use
mdupdate=pd.DataFrame(mean_frc)

# make the date column from the dataframe
mdupdate['date_scored']=mdupdate.index
md['date_scored']=md.index


# In[5]:

# in this case, we are only doing trading after year 2000, so that we truncate the dataframe, but the copy still has 
# all the dates
md = md[md['date_scored']>datetime.date(2000,5,13)]


# In[6]:

score_dict = dict(zip(md.date_scored, md.score))
# putting the data into dictionary


# In[7]:

# getting the SNP data on these dates
startDate = md['date_scored'].min()
endDate = md['date_scored'].max()
print startDate, endDate
snpData = pandas.io.data.get_data_yahoo("^GSPC", startDate, endDate)


# In[8]:

snpData = snpData.loc[:, ['Adj Close']]
snpData['Date'] = snpData.index
# make a copy for further use to create continuous time series from the snpData
snpData2 = snpData.copy()


# In[9]:

# putting the snp prices to dictionary as well
fullPriceDict = dict(zip(snpData['Date'], snpData['Adj Close']))


# In[10]:

# extract the prices on the date that frc has a release
# if there is frc release on a non-week day(no price data), then put None
priceNeed = {}
for r in score_dict:
    if pd.Timestamp(r) in fullPriceDict:
        priceNeed[r] = fullPriceDict[pd.Timestamp(r)]
    else:
        priceNeed[r] = None


# In[11]:

# put in list of tuples in the form (date, price) and sort them according to the date
price_tup = [(k,v) for k,v in priceNeed.iteritems()]
price_tup = sorted(price_tup, key = lambda x:x[0])


# In[12]:

snpData = pd.DataFrame(data = price_tup, columns=['Date', 'Price'])
snpData.index = snpData['Date']
snpData.head()


# In[13]:

# getting the change in price for SNP

#snpData = snpData.fillna(method='bfill')

# calculate the log returns of snpData
snpData['LogReturns'] = numpy.log(snpData['Price']).diff()
snpData.head()


# In[14]:

# getting the change in FRC score
frcData = md.copy()
frcData['ScoreChange'] = frcData['score'].diff()
frcData = frcData.fillna(0)
frcData.head()


# In[15]:

# merge the two DataFrames
compare = pd.concat([snpData, frcData], axis=1)
compare.dropna(inplace=True)


# In[16]:

# only look at these 4 columns
compare = compare.loc[:, ['Price', 'LogReturns', 'score', 'ScoreChange']]
# getting the Moving average of the FRC score both in the truncated copy and the full copy
mdupdate['Frc Moving Average'] = pd.rolling_mean(mdupdate['score'],window=10, min_periods=1)
# calculating the momentum which is just delta Moving average. 
mdupdate['momentum']=mdupdate['Frc Moving Average'].diff()
compare['Frc Moving Average'] = pd.rolling_mean(compare['score'], window=10, min_periods=1)
compare['momentum']=compare['Frc Moving Average'].diff()


# In[17]:

compare.columns = ['SNP Price', 'LogReturns', 'FRC Score', 'Score Change', 'Frc Moving Average', 'momentum']
compare.corr(method='spearman')


# In[18]:

momentum = compare.loc[:, ['momentum']]
compare['Date'] = compare.index

# putting all the dates and momentum in a dictionary, taking care of the Timestamp vs datetime juggling. 
#dateMomen = dict(zip(compare.Date, compare.momentum))
dateMomen = {pd.Timestamp(k):v for k, v in zip(compare.Date, compare.momentum)}
datePrice = dict(zip(compare.Date, compare['SNP Price'])) # this datePrice only has fed release dates


# In[19]:

snpData2 = snpData2.resample('D', fill_method = 'ffill')
snpData2['Date'] = snpData2.index
datePriceC = dict(zip(snpData2.Date, snpData2['Adj Close'])) # this dict includes the prices from a continuous series
# C for continuous or whatever


# In[20]:

portfolio_balance=[]
# first create two lists for grid search on the quantiles
#TopEnd = [(x , momentum['momentum'].quantile(x/100.0)) for x in range(75, 95, 1)]
#BottomEnd = [(x, momentum['momentum'].quantile(x/100.0)) for x in range(5, 25, 1)]
#ParaTuple=[]

# Here we are only manipulating the values but not the quantiles

#TopEnd=[x/100.0 for x in range(0,30,2)]
#BottomEnd=[x/100.0 for x in range(0,-30,-2)]


# In[21]:


#for i in range(len(TopEnd)):
#    for j in range(len(BottomEnd)):
#        ParaTuple.append((TopEnd[i],BottomEnd[j]))
i=0
ParaTuple=[]
TopEnd = [x/100.0 for x in range(75, 95, 1)]
BottomEnd = [x/100.0 for x in range(5, 25, 1)]
for i in range(len(TopEnd)):
    for j in range(len(BottomEnd)):
        ParaTuple.append((TopEnd[i],BottomEnd[j]))
#, mdupdate['momentum'].quantile(x/100.0))
def momentum_mag_trading(date, dictM, Parameters, dframe):
    '''Make trading decsions on a daily basis, want to buy (return 1) if the momentum of the day is higher than some
    cut off value, and sell/short (return -1) if the momentum of the day is lower than some cut-off value. Do nothing
    otherwise.
    
    input: date --> a single date
           dictM --> a dictionary of date:momentum
           parameters --> the quantile/percentile to use in this iteration
           deframe --> the datafram to get the percentile data from
    '''
    if dictM[date] > dframe['momentum'].quantile(Parameters[0]):
        #print 'Take a long position on {0}'.format(date)
        return 1
    elif dictM[date] < dframe['momentum'].quantile(Parameters[1]):
        #print 'Take a short position on {0}'.format(date)
        return -1
    else:
        #print 'No position on {0}'.format(date)
        return 0
    
def isFedDay(date, fedDates):
    """Test whether a certain day is a fed/frc release day
    """
    if date in fedDates:
        return True
    else:
        return False

def backTesting(datels, dictP, dictM, Parameters=(0.93, 0.06), startMoney=10000):
    '''backtesting a a set of previous dates
    
    datels --> a list of CONTINUOUS dates to track portfolio balance
    '''
    shorts=0
    long_count = 0
    Money = startMoney
    for d in datels:
        if isFedDay(d, dictM): #for a fed release day, do trading, track portfolio balance
            
            # for each day, we have to recalculate the momentum percentile based on new data
            mdnaive=mdupdate[mdupdate['date_scored'] < d.to_datetime().date()]
            
            if  Money+(long_count-shorts)*dictP[d] > 0 :
                # we only trade when our portfolio value is positive
                indicator = momentum_mag_trading(d, dictM, Parameters,mdnaive)
                if indicator == 1:
                    Money=Money-shorts*dictP[d]
                    #print 'repurchased {0} shorts at {1}.'.format(shorts, dictP[d])
                    shorts=0
                    if Money> dictP[d]:
                        # We buy whatever number of shares our cash allows us to buy
                        long_count= long_count + math.trunc(Money/dictP[d])
                        Money= Money - math.trunc(Money/dictP[d])*dictP[d]
                    else:
                        pass
                        #print 'No sufficient money on {0}'.format(d) 
       
                elif indicator == -1:
                    Money = Money + long_count*dictP[d]
                #print 'Sold {0} stocks at {1}'.format(long_count, dictP[d])
                    long_count = 0
                    if shorts < 10:
                        # We simply arbitrarily limit the number of shorts to 10
                        shorts= shorts + math.trunc(Money/dictP[d])
                        Money = Money + math.trunc(Money/dictP[d])*dictP[d]
                    #print 'Made a short at {1}. Now have {0} shorts that needs to be paid'.format(shorts, dictP[d])
                    #Money = Money +shorts*dictP[d] previous error
                else:
                    pass
            else: 
                print "We have no money in the bank. Call us Mike Tyson"
        
            #print 'We have {0} in hand'.format(Money)
            portfolio_balance.append(Money+(long_count-shorts)*dictP[d])
            
        else:  # if not a fed release date, only track the portfolio balance
            portfolio_balance.append(Money+(long_count-shorts)*dictP[d])
            
    return 'We have {0} dollars, {1} in stock assest, and {2} in shorts that need to be paid'.format(Money,long_count*dictP[datels[len(datels)-1]],shorts)


# In[22]:

dates = []
#for d in compare['Date'][:]:
for d in snpData2['Date'][:]:
    # now iterating through a continuous series of dates
    dates.append(d)


# In[23]:

print backTesting(dates, datePriceC, dateMomen)


# In[24]:

plt.plot(dates, portfolio_balance)


# In[25]:

#max(portfolio_balance, key=lambda x: x[1])


# In[26]:

#sorted(portfolio_balance ,key = lambda x: x[1])


# In[ ]:



