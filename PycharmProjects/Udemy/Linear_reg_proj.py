import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers")

#print(df.head())
#print(df.info())
#print(df.describe())

#sns.jointplot(df['Time on Website'], df['Yearly Amount Spent'])
#sns.jointplot(df['Time on App'], df['Yearly Amount Spent'])
#sns.jointplot(df['Time on Website'], df['Yearly Amount Spent'], kind='hex')
#sns.pairplot(df)
#sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = df)
#plt.show()

#print(df.columns)

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df[['Yearly Amount Spent']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

lm = LinearRegression()
lm.fit(X_train, y_train)

#print(lm.coef_)

predictions = lm.predict(X_test)

#sns.jointplot(y_test, predictions)
#sns.distplot(y_test-predictions)
#plt.show()

#print('MAE ', metrics.mean_absolute_error(y_test, predictions))
#print('MSE ', metrics.mean_squared_error(y_test, predictions))
#print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#sns.distplot(y_test-predictions, bins=100)
#plt.show()

coeffecients = pd.DataFrame(lm.coef_, columns = X.columns)
print(coeffecients)