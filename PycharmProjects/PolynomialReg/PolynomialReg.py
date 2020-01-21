import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
n_samples = 100

X = np.linspace(0, 10, 100)
rng = np.random.randn(n_samples) * 100

y =X **3 + rng + 100

'''plt.figure(figsize=(10,8))
plt.scatter(X, y)
plt.show()'''

'''Linear modelling'''

'''best = 0
for _ in range(50) :
    model = LinearRegression()
    model.fit(X.reshape(-1,1), y)

    if best < model.score(X.reshape(-1,1), y) :
        best = model.score(X.reshape(-1,1), y)
        with open("linearmodel.pickle", "wb") as f :
            pickle.dump(model, f)
print(best)'''

model_saved = open("linearmodel.pickle", "rb")
linear = pickle.load(model_saved)

lin_predict = linear.predict(X.reshape(-1,1))
acc = linear.score(X.reshape(-1,1), y)
print(acc)
plt.scatter(X,y)
plt.plot(X, lin_predict)
plt.show()

'''Polynomial Modelling'''
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X.reshape(-1,1))

'''best = 0
for _ in range(50) :
    model = LinearRegression()
    model.fit(X_poly, y)

    if best < model.score(X_poly, y) :
        best = model.score(X_poly, y)
        with open("polymodel.pickle", "wb") as f :
            pickle.dump(model, f)

print(best)'''
model_saved2 = open("polymodel.pickle", "rb")
polynomial = pickle.load(model_saved2)

poly_predict = polynomial.predict(X_poly)
acu = polynomial.score(X_poly, y)
print(acu)

plt.figure(figsize= (10, 8))
plt.scatter(X, y)
plt.plot(X, poly_predict)
plt.show()