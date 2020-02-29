import nltk

#nltk.download_shell()

messages = [line.rstrip() for line in open('/home/srijan/Downloads/SMSSpamCollection')]
'''
for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')
'''
import pandas as pd

messages = pd.read_csv('/home/srijan/Downloads/SMSSpamCollection', sep='\t', names=['label','messages'])

#print(messages.describe())
#print(messages.groupby('label').describe())

messages['length'] = messages['messages'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns

#sns.distplot(messages['length'], bins=150, kde=False)
#plt.show()

#print(messages.describe())
#print(messages[messages['length']==910]['messages'].iloc[0])

#g = sns.FacetGrid(messages, sharey=True, col='length', hue='label')
#messages.hist(column='length', by='label', bins=60, figsize=(10,6))
#plt.show()

#exploratory data analysis over. Now actual processing starts

import string

mess = 'Sample Message! Notice: It has punctuatuin, colon and period.'
nopunc = [c for c in mess if c not in string.punctuation]
#print(nopunc)

from nltk.corpus import stopwords
#print(stopwords.words('english'))

nopunc = ''.join(nopunc) #the enclosed characters in ''.join are the joining part. here we've left it empty
#print(nopunc)

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#print(clean_mess)

def text_process(mess):
    '''
    this is process of tokenization of a string
    1. remove punc
    2. remove stop words
    3. return list of cleaned words
    '''
    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#print(text_process(mess))

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['messages'])
#print(len(bow_transformer.vocabulary_))


mess4 = messages['messages'][3]
bow4 = bow_transformer.transform([mess4])
#print(bow4)
#print(bow4.shape)
#print(bow_transformer.get_feature_names()[*column you want to check*]) # to see which fearure is here


messages_bow = bow_transformer.transform(messages['messages'])
'''
print("Shape of Sparse matrix: ", messages_bow.shape)
print("Non zero occurances: ",messages_bow.nnz)
print("Zero occurances: ",messages_bow.shape[0]*messages_bow.shape[1]-messages_bow.nnz)
'''

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)

#tfidf4 = tfidf_transformer.transform(bow4)
#print(tfidf4)

# to check inverse document frequency
#print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

messages_tfidf = tfidf_transformer.transform(messages_bow) # Vectorication finished

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

#print('Predicted: ', spam_detect_model.predict(tfidf_transformer.transform(bow4))[0], ' Actual: ', messages['label'][3])

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['messages'], messages['label'], test_size = 0.3)

from sklearn.pipeline import Pipeline
'''
insted of doing all above steps again, we use the pipeline method
'''
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),('classifier', MultinomialNB())])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print('Using Naive Bayes Classifier\n',classification_report(label_test, predictions))

from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),('classifier', RandomForestClassifier())])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print('Using Random Forest Classifier\n',classification_report(label_test, predictions))

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),('classifier', KNeighborsClassifier())])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print('Using K Neighbors Classifier\n',classification_report(label_test, predictions))

from sklearn.svm import SVC
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),('classifier', SVC())])
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print('Using SVC\n',classification_report(label_test, predictions))