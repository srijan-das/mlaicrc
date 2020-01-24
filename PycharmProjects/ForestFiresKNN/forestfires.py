import pandas as pd
#import numpy as np
import sklearn
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv("forestfires.csv", sep=",")

pre = preprocessing.LabelEncoder()
month = pre.fit_transform(data['month'])
day = pre.fit_transform(data['day'])
x = pre.fit_transform(data['X'])
y = pre.fit_transform(data['Y'])
FFMC = pre.fit_transform(data['FFMC'])
DMC = pre.fit_transform(data['DMC'])
DC = pre.fit_transform(data['DC'])
ISI = pre.fit_transform(data['ISI'])
temp = pre.fit_transform(data['temp'])
RH = pre.fit_transform(data['RH'])
wind = pre.fit_transform(data['wind'])
rain = pre.fit_transform(data['rain'])
area = pre.fit_transform(data['area'])


y = list(area)
x = list(zip(x, y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

'''knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
acc = knn_model.score(x_test, y_test)
print(acc)'''

clf = svm.SVC(kernel='linear', C = 2)
clf.fit(x_train, y_train)
sacc = clf.score(x_test, y_test)
print(sacc)