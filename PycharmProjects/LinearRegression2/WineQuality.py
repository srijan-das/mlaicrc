import tensorflow
import sklearn
from sklearn import linear_model
from sklearn import model_selection
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style
import numpy as np
import seaborn as seaborn
import pickle

data = pd.read_csv("winequality.csv", sep=",")

x = np.array(data.drop(['quality'], 1))
y = np.array(data['quality'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(50) :
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    if acc > best :
        best = acc
        with open("qualitypredictor.pickle", "wb") as f :
            pickle.dump(model,f)

print(best)'''

model_saved = open("qualitypredictor.pickle", "rb")
predictor = pickle.load(model_saved)

predictions = predictor.predict(x_test)

print(predictor.score(x_test, y_test), "\n")

for i in range(len(predictions)) :
    print(predictions[i], "  ",  y_test[i])

p = 'quality'
q = 'alcohol'
style.use("seaborn")
pyplot.scatter(data[p], data[q])
pyplot.xlabel(p)
pyplot.ylabel(q)
pyplot.show()

'''print(pyplot.style.available)'''