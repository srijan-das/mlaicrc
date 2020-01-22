import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data", sep = ",")
'''print(data.head())'''

pre = preprocessing.LabelEncoder()
buying = pre.fit_transform(data['buying'])
maint = pre.fit_transform(data['maint'])
lug_boot = pre.fit_transform(data['lug_boot'])
safety = pre.fit_transform(data['safety'])
persons = pre.fit_transform(data['persons'])
door = pre.fit_transform(data['door'])
cls = pre.fit_transform(data['class'])

predict = "class"

y = list(cls)
x = list(zip(buying, maint, door, persons, lug_boot, safety))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
acc = model.score(x_test, y_test)
print(acc)

names = ["acc", "unacc", "good", "vgood"]

for i in range(len(predictions)) :
    print("Data :", x_test[i], " Predicted: ", names[predictions[i]], " Actual : ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 7, True)
    print("N :", n)
