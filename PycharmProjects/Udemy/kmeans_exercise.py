import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs

data = make_blobs(n_samples= 200, n_features= 2, centers= 4, cluster_std= 1.75, random_state=101)
'''
sns.scatterplot(data[0][:,0], data[0][:,1], hue= data[1], palette='rainbow')
plt.show()
'''

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

#Visualization using inbuilt. NOT WORKING

fig , (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c = kmeans.labels_, cmap = 'rainbow')

ax2.set_title('Original')
ax1.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')

plt.show()

#Visualization using Seaborn, NOT WORKING YET
'''
df =pd.DataFrame(data = data[0], columns=['Col 1','Col 2'])
ogi = pd.DataFrame(data = data[1], columns = ['Original'])
kmeans = pd.DataFrame(data = kmeans.labels_, columns=['K Means'])
obj = [df, ogi, kmeans]
dataframe = pd.concat(obj, axis=1)
print(dataframe)

g = sns.FacetGrid(dataframe,)
'''