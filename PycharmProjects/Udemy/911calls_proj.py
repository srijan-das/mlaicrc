import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv')
#print(data.head())

#print(df['zip'].value_counts().head(5))
#print(df['twp'].value_counts().head(5))
#print(df['title'].nunique())

def reasoner(x):
    if type(x) == str :
        temp = x.split(':')
        return temp[0]

df['reason'] = df['title'].apply(reasoner)
#print(df['reason'].value_counts().head(5))

#sns.countplot(x='reason', data = df)
#plt.show()

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['hour'] = df['timeStamp'].apply(lambda x : x.hour)
df['month'] = df['timeStamp'].apply(lambda x : x.month)
df['day'] = df['timeStamp'].apply(lambda x : x.dayofweek)

dmap = {0 : 'Mon', 1 : 'Tues', 2 :'Wed', 3 : 'Thurs', 4 : 'Fri', 5 : 'Sat', 6 : 'Sun'}
df.day = [dmap[item] for item in df.day]
#print(df['day'].value_counts())

#sns.countplot(x='month', hue='reason', data=df)
#plt.show()

byMonth = df.groupby('month').count()

#print(byMonth)
#byMonth['title'].plot()
#sns.lmplot(x='month', y='title', data=byMonth.reset_index())
#plt.show()

df['date'] = pd.to_datetime(df['timeStamp'])
df['date'] = df['date'].apply(lambda x : x.date())

#df.groupby('date').count()['twp'].plot()
#plt.tight_layout()
#plt.show()

#criteria = 'Traffic'
#df[df['reason'] == criteria].groupby('date').count()['twp'].plot()
#plt.title(criteria)
#plt.tight_layout()
#plt.show()
'''
ptdf = df[['hour', 'day']]
ptdf1 = ptdf.groupby(by = ['day', 'hour']).count()['reason'].unstack()
print(ptdf1.head())'''

dayHour = df.groupby(by=['day','hour']).count()['reason'].unstack()
#print(dayHour.head())

#sns.heatmap(dayHour)
#plt.show()

#sns.clustermap(dayHour)
#plt.show()

dayMonth = df.groupby(by=['day', 'month']).count()['reason'].unstack()
sns.heatmap(dayMonth)
plt.show()