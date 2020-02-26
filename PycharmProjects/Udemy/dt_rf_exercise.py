import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv")

#print(df.head())
#print(df.info())

#sns.pairplot(df, hue = 'Kyphosis')
#plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

pred = tree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators= 250)

forest.fit(X_train, y_train)
predrf = forest.predict(X_test)

print('\n')
print(confusion_matrix(y_test, predrf))
print('\n')
print(classification_report(y_test, predrf))