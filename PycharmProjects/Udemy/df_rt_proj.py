import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv")

#print(df.columns)
'''
plt.figure(figsize = (10,6))
df[df['credit.policy']==1]['fico'].hist(alpha = 0.5, color = 'blue', bins = 30, label = 'Credit approved')
df[df['credit.policy']==0]['fico'].hist(alpha = 0.5, color = 'red', bins = 30, label = 'Credit not approved')
plt.legend()
plt.xlabel('fico')
plt.show()

plt.figure(figsize = (10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha = 0.5, color = 'blue', bins = 30, label = 'Not paid')
df[df['not.fully.paid']==0]['fico'].hist(alpha = 0.3, color = 'red', bins = 30, label = 'Paid')
plt.legend()
plt.xlabel('fico')
plt.show()

sns.countplot(data = df, x='purpose', hue = 'not.fully.paid' )
plt.show()

sns.jointplot(data = df, x = 'fico', y = 'int.rate')
plt.show()

sns.lmplot(data = df, x = 'fico', y = 'int.rate', hue = 'not.fully.paid', col = 'not.fully.paid')
plt.show()
'''

purpose = ['purpose']
final_data = pd.get_dummies(df, columns = purpose, drop_first = True)
#print(final_data.info())

from sklearn.model_selection import train_test_split

X = df.drop(['not.fully.paid','purpose'], axis = 1)
y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
treepreds = dtree.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forestpred = forest.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("For Decision Tree \n")
print(classification_report(y_test, treepreds))
print('\n')
print(confusion_matrix(y_test, treepreds))

print("For Random Forests Tree \n")
print(classification_report(y_test, forestpred))
print('\n')
print(confusion_matrix(y_test, forestpred))