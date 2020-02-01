from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chart_studio.plotly as iplot
import cufflinks as cf

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

BAC_data = data.get_data_tiingo('BAC', api_key='66d0fd8f718e53b4dd3b76fce1d34c53ee8d7dc1', start = start, end = end)
C_data = data.get_data_tiingo('C', api_key='66d0fd8f718e53b4dd3b76fce1d34c53ee8d7dc1', start = start, end = end)
GS_data = data.get_data_tiingo('GS', api_key='66d0fd8f718e53b4dd3b76fce1d34c53ee8d7dc1', start = start, end = end)
MS_data = data.get_data_tiingo('MS', api_key='66d0fd8f718e53b4dd3b76fce1d34c53ee8d7dc1', start = start, end = end)
WFC_data = data.get_data_tiingo('WFC', api_key='66d0fd8f718e53b4dd3b76fce1d34c53ee8d7dc1', start = start, end = end)

BAC = BAC_data.xs('BAC')
C = C_data.xs('C')
GS = GS_data.xs('GS')
MS = MS_data.xs('MS')
WFC = WFC_data.xs('WFC')

tickers = 'BAC C GS MS WFC'.split()

bank_data = pd.concat([BAC, C, GS, MS, WFC], axis = 1, keys = tickers)
bank_data.columns.names = ['Bank Ticker','Stock Info']

returns = pd.DataFrame()

for tick in tickers :
    returns[tick+' returns'] = bank_data[tick]['close'].pct_change()

sns.pairplot(returns[1:])
plt.show()

print(returns.idxmin())

print(returns.idxmax())

print(returns.std())

print(returns.ix['2015-01-01' : '2015-12-31'].std())

sns.distplot(returns.ix['2015-01-01' : '2015-12-31']['MS returns'], color = 'green', bins = 100)
plt.show()

sns.distplot(returns.ix['2008-01-01' : '2008-12-31']['C returns'], color = 'green', bins = 100)
plt.show()

for tick in tickers:
    bank_data[tick]['close'].plot(figsize=(12,4), label=tick)
plt.legend()
plt.show()

bank_data.xs(key='close', axis = 1, level = 'Stock Info').plot(figsize=(12, 4))
plt.show()

plt.figure(figsize = (12,6))
BAC['close'].ix['2008-01-01' : '2008-12-31'].rolling(window = 30).mean().plot(label = '30 Day Avg')
BAC['close'].ix['2008-01-01' : '2008-12-31'].plot(label='Daily Close')
plt.title('Bank Of America Closing Prices, 2008')
plt.show()

sns.heatmap(bank_data.xs(key = 'close', axis = 1, level = 'Stock Info').corr(), annot = True, cmap = 'magma')
plt.show()