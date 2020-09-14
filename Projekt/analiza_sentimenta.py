import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from nltk import FreqDist, classify, NaiveBayesClassifier
import pandas as pd
from os import system, name
from time import sleep

import re, string, random
#funkcija za ciscenje ekrana
def Clear():
    if name == 'nt':
        _ = system('cls')

#funkcija za ciscenje podataka
def CleanUpData(data):
    clean_data = []
    for i in range(data.shape[0]):
        print(i,"/",data.shape[0])
        #Clear()
        soup = BeautifulSoup(data.iloc[i], "html.parser")
        entry = soup.get_text()
        entry = re.sub('\[[^]]*\]', ' ', entry)
        entry = re.sub('[^a-zA-Z]', ' ', entry)
        entry = entry.lower()
        entry = entry.split()
        entry = [word for word in entry if not word in set (stopwords.words('english'))]
        lem = WordNetLemmatizer()
        entry = [lem.lemmatize(word, pos="v") for word in entry]
        entry = ' '.join(entry)
        clean_data.append(entry)
    return clean_data

#podjela podataka
movie_data =pd.read_csv("IMDB Dataset.csv")
movie_data = movie_data[:1000]
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(movie_data['review'], movie_data['sentiment'], test_size=0.25, random_state=42)

train = movie_data[:500]
test = movie_data[500:]
corpus_train = []
corpus_test = []

train_sentiments = (train_sentiments.replace({'positive':1,'negative':0})).values
test_sentiments = (test_sentiments.replace({'positive':1,'negative':0})).values

corpus_train = CleanUpData(train_reviews)
corpus_test = CleanUpData(test_reviews)
#corpus_train = CleanUpData(train)
#corpus_test = CleanUpData(test)

count_vec = CountVectorizer(ngram_range=(1, 3), binary=False)
count_vec_train = count_vec.fit_transform(corpus_train)
count_vec_test = count_vec.transform(corpus_test)



linear_svc_count = LinearSVC(C=0.5, random_state=42, max_iter=5000)
linear_svc_count.fit(count_vec_train, train_sentiments)

predict_count = linear_svc_count.predict(count_vec_test)



print("Classification Report: \n", classification_report(test_sentiments, predict_count,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_sentiments, predict_count))
print("Accuracy: \n", accuracy_score(test_sentiments, predict_count))

