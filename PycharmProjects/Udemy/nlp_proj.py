import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv("/home/srijan/PycharmProjects/mlaicrc/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv")
'''
print(yelp.head())
print(yelp.describe())
print(yelp.info())
print(yelp.columns)
'''

yelp['text length'] = yelp['text'].apply(len)
'''
g = sns.FacetGrid(yelp, 'stars')
g.map(sns.distplot, 'text length')
plt.show()
'''
'''
sns.boxplot(data=yelp, x='stars', y='text length')
plt.show()
'''
'''
sns.countplot(data=yelp, x='stars')
plt.show()
'''

mean_stars = yelp.groupby(['stars']).mean()
#print(mean_stars.head())

'''
print(mean_stars.corr())
sns.heatmap(mean_stars.corr(), cmap='magma', annot=True)
plt.show()
'''
yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
#print(yelp_class.head())

from sklearn.model_selection import train_test_split

X = yelp_class['text']
y = yelp_class['stars']
#print(y.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#Without TF-IDF of BOW

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

pred = nb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#With TF_IDF of BOW

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()),('tfidf', TfidfTransformer()),('classification', MultinomialNB())])
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

# Doing some data cleaning

from nltk.corpus import stopwords
import string

def text_processor(text) :
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    final = [word for word in no_punc.split() if word not in stopwords.words('english')]
    return final

#print(text_processor('Hey there! I am Srijan. Nice to meet you. Martin Voh Ludenberg: havent seen you'))

pipe = Pipeline([('bow', CountVectorizer(analyzer=text_processor)),('tfidf', TfidfTransformer()),('classification', MultinomialNB())])
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))