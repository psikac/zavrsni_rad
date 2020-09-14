import pandas as pd
from nltk import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re,string,random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv("IMDB Dataset.csv")
#dataset = dataset[:25000]

print(dataset.head())
print(dataset.info())

#funkcija za ciscenje podataka
def CleanUpData(data):
    clean_data = []
    for i in range(data.shape[0]):
        print(i,"/",data.shape[0])
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
#postavljanje pocetnih vrijednosti i rasporedivanje podataka
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(dataset['review'], dataset['sentiment'], test_size=0.25, random_state=42)
corpus_train = []
corpus_test = []


train_sentiments = (train_sentiments.replace({'positive': 1, 'negative': 0})).values
test_sentiments  = (test_sentiments.replace({'positive': 1, 'negative': 0})).values



corpus_train = CleanUpData(train_reviews)
corpus_test = CleanUpData(test_reviews)


#kreiranje testnih podataka za direktan pregled rezultata


dataset_predict = test_reviews.copy()
dataset_predict = pd.DataFrame(dataset_predict)
dataset_predict.columns = ['review']
dataset_predict = dataset_predict.reset_index()
dataset_predict = dataset_predict.drop(['index'], axis=1)

test_actual_label = test_sentiments.copy()
test_actual_label = pd.DataFrame(test_actual_label)
test_actual_label.columns = ['sentiment']
test_actual_label['sentiment'] = test_actual_label['sentiment'].replace({1: 'positive', 0: 'negative'})

"""

tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))

tfidf_vec_train = tfidf_vec.fit_transform(corpus_train)
tfidf_vec_test = tfidf_vec.transform(corpus_test)

count_vec = CountVectorizer(ngram_range=(1, 2), binary=False)

count_vec_train = count_vec.fit_transform(corpus_train)
count_vec_test = count_vec.transform(corpus_test)


linear_svc_tfidf = LinearSVC(C=0.5, random_state=42)
linear_svc_tfidf.fit(tfidf_vec_train, train_sentiments)

#TF-IDF & Linear SVC
predict_svc_tfidf = linear_svc_tfidf.predict(tfidf_vec_test)

print("Classification Report: \n", classification_report(test_sentiments, predict_svc_tfidf,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_sentiments, predict_svc_tfidf))
print("Accuracy: \n", accuracy_score(test_sentiments, predict_svc_tfidf))


test_predicted_label = predict_svc_tfidf.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'positive', 0: 'negative'})


test_result = pd.concat([dataset_predict, test_actual_label,test_predicted_label], axis=1)
print(test_result.head())


#CountVectionizer & Linear SVC
linear_svc_count = LinearSVC(C=0.5, random_state=42)
linear_svc_count.fit(count_vec_train, train_sentiments)

predict_svc_count = linear_svc_count.predict(count_vec_test)

print("Classification Report: \n", classification_report(test_sentiments, predict_svc_count,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_sentiments, predict_svc_count))
print("Accuracy: \n", accuracy_score(test_sentiments, predict_svc_count))

test_predicted_label = predict_svc_count.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'positive', 0: 'negative'})


test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
print(test_result.head())

#TF-IDF & Multinomial Naive Bayes

tfidf_vec_NB = TfidfVectorizer(ngram_range=(1, 2))
tfidf_vec_train_NB = tfidf_vec_NB.fit_transform(corpus_train)

tfidf_vec_test_NB = tfidf_vec_NB.transform(corpus_test)



multi_clf = MultinomialNB()
multi_clf.fit(tfidf_vec_train_NB, train_sentiments)

predict_NB = multi_clf.predict(tfidf_vec_test_NB)

print("Classification Report: \n", classification_report(test_sentiments, predict_NB,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_sentiments, predict_NB))
print("Accuracy: \n", accuracy_score(test_sentiments, predict_NB))

test_predicted_label = predict_NB.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'positive', 0: 'negative'})

test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
print(test_result.head())
"""
#CountVectionizer & Multinomial Naive Bayes


count_vec_NB = CountVectorizer(ngram_range=(1, 2), binary=False)
count_vec_train_NB = count_vec_NB.fit_transform(corpus_train)
count_vec_test_NB = count_vec_NB.transform(corpus_test)


multi_clf_count = MultinomialNB()
multi_clf_count.fit(count_vec_train_NB, train_sentiments)

predict_NB_count = multi_clf_count.predict(count_vec_test_NB)


print("Classification Report: \n", classification_report(test_sentiments, predict_NB_count,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(test_sentiments, predict_NB_count))
print("Accuracy: \n", accuracy_score(test_sentiments, predict_NB_count))

test_predicted_label = predict_NB_count.copy()
test_predicted_label = pd.DataFrame(test_predicted_label)
test_predicted_label.columns = ['predicted_sentiment']
test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace({1: 'positive', 0: 'negative'})

test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
print(test_result.head())
