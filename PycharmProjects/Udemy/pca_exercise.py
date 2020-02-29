import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(type(cancer))
#print(cancer.keys())
#print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

#print(df.head())
#print(df.info())

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df)
scaled_data = scale.transform(df)
#print(scaled_data.shape)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
#print(x_pca.shape)

ar1 = np.array(x_pca[:,0])
ar2 = np.array(x_pca[:,1])

c1 = pd.DataFrame(data=ar1, columns=['Principal Component 1'])
c2 = pd.DataFrame(data=ar2, columns=['Principal Component 2'])
c3 = pd.DataFrame(data=cancer['target'], columns=['Malignant'])
obj = [c1, c2, c3]
table = pd.concat(obj, axis=1)
print(table.shape)

plt.figure(figsize = (8,6))
sns.scatterplot(data=table, x='Principal Component 1' , y='Principal Component 2', hue='Malignant')
plt.show()

df_pc = pd.DataFrame(data=pca.components_, columns=cancer['feature_names'])
sns.heatmap(df_pc, cmap='plasma')
plt.show()