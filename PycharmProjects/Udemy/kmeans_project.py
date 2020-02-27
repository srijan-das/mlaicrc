import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

college_data = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data")
'''
print(college_data.describe())
print(college_data.info())
print(college_data.head())
print(college_data.columns)

sns.scatterplot(data = college_data, x='Apps', y='Accept', hue='Private', alpha=0.8)
plt.show()
'''

from sklearn.cluster import KMeans

#elbow method for optimal k
'''
distortions = []
data = college_data.drop(['Private','Unnamed: 0'], axis = 1)
for k in range(1,10) :
    model = KMeans(n_clusters=k)
    model.fit(data)
    distortions.append(model.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,10), distortions, marker = 'o')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('Elbow Method for optimal K')
plt.show()
'''
model = KMeans(n_clusters=2)
model.fit(college_data.drop(['Private','Unnamed: 0'], axis = 1))
lab = pd.DataFrame(model.labels_, columns=['Labels'])
obj = [college_data, lab]
data_final = pd.concat(obj, axis = 1)

sns.scatterplot(data=data_final, x='Outstate', y = 'F.Undergrad', hue = 'Private', palette='rainbow')
plt.show()
sns.scatterplot(data=data_final, x='Outstate', y = 'F.Undergrad', hue = 'Labels', palette='rainbow')
plt.show()
