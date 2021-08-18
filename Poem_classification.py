from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import pandas as pd
from os import walk
import numpy as np
import pickle
import hazm
import nltk
import re

# define the path of dataset
path = '/home/hakim/Documents/semester 8/IR/HW_4/Data/Poems-final'
my_regex = '[^ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌\s]+|[۰۱۲۳۴۵۶۷۸۹]+|[\\n]'

# open stopwords file
stop_words_file = open('/home/hakim/Documents/semester 8/IR/HW_4/Data/PersianPoemsData/Stopwords/Stopwords', 'r')
stop_words = stop_words_file.read()

# create a dictionary to save poem and poets that been read from dataset path
dataset_dict = {}
poets = []
for root, dirs, files in walk(path):
    for dir in dirs:
        dataset_dict[dir] = []
        poets.append(dir)
    break

# poets = ['moulavi', 'saadi', 'hafez', 'khayyam']

# open each poeter file and their poem in the dataset_dict
for poet in poets:
    for root, dirs, files in walk(path+'/'+poet):
        for file in files:
            file = open(root+'/'+file)
            text = file.read()
            text = re.sub(my_regex, ' ', text)
            normalizer = hazm.Normalizer()
            text = normalizer.normalize(text)
            stemmer = hazm.Stemmer()
            text = stemmer.stem(text)
            if text != ' ':
                dataset_dict[poet].append(text)
            file.close()

# print log for showing an example of dataset_dict
print(dataset_dict.keys())
print(dataset_dict[poets[2]][1])

# creating a dataframe with the dataset_dict
dataset = {'poem' : [],
        'poet' : []}

for i in dataset_dict:
    dataset['poem'] += (dataset_dict[i])
    
for i in dataset_dict:
    for _ in range(len(dataset_dict[i])):
        dataset['poet'].append(i)
        
df = pd.DataFrame(dataset)
print(df)

# add the class column to the made datafraem
columns = ['poet', 'poem']
df = df[columns]
df = df[pd.notnull(df['poet'])]
df.columns = ['poet', 'poem']
df['class'] = df['poet'].factorize()[0]
temp = df[['poem', 'class']].drop_duplicates()
class_id_df = temp.sort_values('class')
class_to_id = dict(class_id_df.values)
id_to_class = dict(class_id_df[['class', 'poem']].values)
print(df)


# crate the vectorizer model and fit it with poems
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=set(stop_words))
features = tfidf.fit_transform(df['poem']).toarray()
labels = df['class']

# use train test spiter functoin to make train and test dataframe
X_train, X_test, y_train, y_test = train_test_split(df['poem'], df['poet'], test_size=0.30, random_state = 42)

# print a log to show an example of the train dataframe
print('v'*50)
print(X_train.describe)

# create and training the model using Multinominal Naive Bayes class of sklearn library
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(X_train)
model = TfidfTransformer()
X_train_tfidf = model.fit_transform(X_train_counts)
trained_model = MultinomialNB().fit(X_train_tfidf, y_train)

# use a sample poem line to test the predictor model
poem_to_predict = ['بشنو از نی چون حکایت می‌کند']
predict = trained_model.predict(count_vector.transform(poem_to_predict))
print('*'*50)
print('Prediction for: ', poem_to_predict, 'is ---> ', predict[0])

# print prediction result
print('*'*50)
print('Predictions with naive bayes: ')
predict = trained_model.predict(count_vector.transform(X_test))
print(predict)

# show naive bayes precision
print('Naive bayes precision:\n', classification_report(y_test, predict))

# create and training the model using knn1 algorithm of sklearn library
knn_1_model = KNeighborsClassifier(n_neighbors=1)
knn_1_model.fit(X_train_tfidf, y_train)
knn_1_predictions = knn_1_model.predict(count_vector.transform(X_test)) 

print('*'*50)
print('Predictions with 1 knn: ')
print(knn_1_predictions)

# show knn1 precision
print('knn1 precision:\n', classification_report(y_test, knn_1_predictions))

# create and training the model using knn3 algorithm of sklearn library
knn_3_model = KNeighborsClassifier(n_neighbors=3)
knn_3_model.fit(X_train_tfidf, y_train)
knn_3_predictions = knn_3_model.predict(count_vector.transform(X_test)) 

print('*'*50)
print('Predictions with 3 knn: ')
print(knn_3_predictions)

# show knn3 precision
print('knn3 precision:\n', classification_report(y_test, knn_3_predictions))


# create and training the model using knn5 algorithm of sklearn library
knn_5_model = KNeighborsClassifier(n_neighbors=3)
knn_5_model.fit(X_train_tfidf, y_train)
knn_5_predictions = knn_5_model.predict(count_vector.transform(X_test)) 

print('*'*50)
print('Predictions with 5 knn: ')
print(knn_5_predictions)

# show knn5 precision
print('knn5 precision:\n', classification_report(y_test, knn_1_predictions))


# create and training the model using linear-svc of sklearn library
svc_model = LinearSVC()
svc_model.fit(X_train_tfidf, y_train)
svc_predict = svc_model.predict(count_vector.transform(X_test))

print('*'*50)
print('Predictions linear-svc knn: ')
print(svc_predict)

# show linear-svc precision
print('linear-svc precision:\n', classification_report(y_test, knn_1_predictions))

