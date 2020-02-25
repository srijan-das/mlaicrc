import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

ad_data = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv")
#print(ad_data.columns)

#print(ad_data.info)
#print(ad_data.describe())

#sns.distplot(ad_data['Age'], kde = False, bins = 50)
#plt.show()

ad_data.drop(['Timestamp', 'Ad Topic Line', 'Country', 'City'], axis = 1, inplace = True)
#print(ad_data.info())

#sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Clicked on Ad'], kind = 'kde')
#sns.pairplot(ad_data, hue = 'Clicked on Ad')
#sns.heatmap(ad_data.isnull(), yticklabels= False, cbar = False, cmap = 'magma')
#plt.show()

#print(ad_data.columns)
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

from sklearn.linear_model import LogisticRegression

logmod = LogisticRegression()
logmod.fit(x_train, y_train)

predictions = logmod.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

sns.distplot(y_test-predictions, bins = 100, kde = False)
plt.show()