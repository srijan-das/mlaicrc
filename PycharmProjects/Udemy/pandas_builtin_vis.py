import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/srijan-das/mlaicrc/master/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df3'

data = pd.read_csv(url)

sns.set_style(style='whitegrid')

#print(data.head())

#data.plot.scatter(x='b', y='a', figsize= (12,4), color='red', s=50)
#plt.show()

#data['a'].plot.hist(alpha = 0.5, bins= 25)
#plt.show()

#data[['a','b']].plot.box()
#plt.show()

#data['d'].plot.kde(c='red', lw = 4, ls='--')
#plt.show()

data.ix[0:30].plot.area(alpha = 0.4)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()