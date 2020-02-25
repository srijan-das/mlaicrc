import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data")

#sns.pairplot(knndata, hue = 'TARGET CLASS')
#plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size = 0.33)

from sklearn.neighbors import KNeighborsClassifier

error_rate =[]

for i in range(1,40) :
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 7)
plt.title("Error rate vs k")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))