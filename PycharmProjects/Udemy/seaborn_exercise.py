import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')

url = 'Downloads/train.csv'
titanic = pd.read_csv(url)

#print(titanic.info())

#sns.jointplot(x='Fare', y='Age', data=titanic)
#plt.show()

#sns.distplot(titanic['Fare'], kde= False, color='red', bins= 100)
#plt.show()

#sns.boxplot(x= 'Pclass', y= 'Age', data= titanic, palette= 'rainbow')
#plt.show()

#sns.swarmplot(x= 'Pclass', y= 'Age', data= titanic)
#plt.show()

#sns.countplot(x= 'Sex', data= titanic)
#plt.show()

#sns.heatmap(titanic.corr(), cmap='coolwarm')
#plt.show()

grid = sns.FacetGrid(titanic, col="Sex")
grid.map(plt.hist, 'Age')
plt.show()