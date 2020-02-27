import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

#print(iris.head())

df = pd.DataFrame(iris)
#print(df.head())
'''
sns.pairplot(df, hue='species')
sns.jointplot(df['sepal_length'], df['sepal_width'], kind='kde')
plt.show()
'''
print(df.info())

X = df.drop('species', axis = 1)
y = df['species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 3)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)
print(grid.best_params_)

from sklearn.metrics import classification_report, confusion_matrix

print("Without Grid Search \n")
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))

print("With Grid Search \n")
print(classification_report(y_test, grid_pred))
print('\n')
print(confusion_matrix(y_test, grid_pred))