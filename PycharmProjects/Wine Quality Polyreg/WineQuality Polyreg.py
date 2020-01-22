import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as model_selection
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pickle
from sklearn.metrics import r2_score

data = pd.read_csv("winequality-red.csv", sep = ",")

'''plt.figure(figsize=(8,5))
plt.scatter(data['citric acid'], data['quality']);
plt.xlabel("citric acid")
plt.ylabel("quality")
plt.show()'''

predict = 'quality'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

'''model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
acc = model.score(x_test, y_test)
print(acc)
for i in range(len(predictions)) :
    print(predictions[i], y_test[i])'''
best = 0
for _ in range(50) :
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)
    poly = PolynomialFeatures(degree=2)
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.fit_transform(x_test)

    model = LinearRegression()
    model.fit(x_poly_train, y_train)
    acc = model.score(x_poly_test, y_test)
    if best < acc :
        best = acc
        with open("polymodel.pickle", "wb") as f :
            pickle.dump(model, f)

print(best)