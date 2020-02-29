import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id','rating', 'time_stamp']

df = pd.read_csv('/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/u.data', sep='\t', names=column_names)

movie_titles = pd.read_csv('/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/Movie_Id_Titles')

df = pd.merge(df, movie_titles, on='item_id')

#print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

#print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
#print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
#print(ratings.head())

#sns.distplot(ratings['num_of_ratings'], kde=False, bins=70)
#plt.show()
#sns.distplot(ratings['rating'], kde=False, bins=70)
#plt.show()
sns.jointplot(data = ratings, x='rating', y='num_of_ratings', alpha=0.5)
plt.show()