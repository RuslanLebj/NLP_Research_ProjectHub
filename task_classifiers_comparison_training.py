import nltk
import pandas as pd
import pymorphy2
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import time
import pickle
import cloudpickle
import os
morph = pymorphy2.MorphAnalyzer()


# Класс для токенизации предложения на слова, привидения к нижнему регистру,
# удаления стоп слов и знаков пунктуации, лемматизации слова.
class WordsPreprocessingTokenizer:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("russian")
        self.punctuation = string.punctuation
        self.tokenizer = nltk.word_tokenize
        self.lemmatizer = lambda token: morph.parse(token)[0].normal_form

    def tokenize(self, sentence):
        words = [self.lemmatizer(token) for token in self.tokenizer(sentence.lower())
                 if token not in self.stopwords and token not in self.punctuation]
        return words


df_training_set = pd.DataFrame({"sentence": [], "class": []})
path_in = "classifier_sets/training_set"
for filename in os.listdir(f'{path_in}'):  # Считываем все предложения
    df_sentences = pd.read_csv(f'{path_in}/{filename}', encoding="windows-1251", sep=";")
    df_training_set = pd.concat([df_training_set, df_sentences], ignore_index=True)
# Перемешиваем элементы выборки
df_training_set = df_training_set.sample(frac=1).reset_index(drop=True)
# Извлекаем элементы выборки
X = df_training_set.iloc[:, 0].values
y = df_training_set.iloc[:, 1].values
# Разделяем выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


# Векторизация и токенизация собственной функцией
tokenizer = WordsPreprocessingTokenizer()
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, token_pattern=None)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# Обучение
svc_clf = SVC().fit(X_train_tfidf, y_train)
lsvc_clf = LinearSVC().fit(X_train_tfidf, y_train)
sgd_clf = SGDClassifier().fit(X_train_tfidf, y_train)
nb_clf = MultinomialNB().fit(X_train_tfidf, y_train)


# Сравнение
print('Метод опорных векторов с RBF ядром')
start_time = time.perf_counter()
preds_svc = svc_clf.predict(X_test_tfidf)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(classification_report(preds_svc, y_test))
print(f"время: {execution_time} секунд")


print('Метод опорных векторов c линейным ядром')
start_time = time.perf_counter()
preds_lsvc = lsvc_clf.predict(X_test_tfidf)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(classification_report(preds_lsvc, y_test))
print(f"время: {execution_time} секунд")


print('Метод опорных векторов (стохастический градиентный спуск)')
start_time = time.perf_counter()
preds_sgd = sgd_clf.predict(X_test_tfidf)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(classification_report(preds_sgd, y_test))
print(f"время: {execution_time} секунд")


print('Наивный Байесовский классификатор')
start_time = time.perf_counter()
preds_nb = nb_clf.predict(X_test_tfidf)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(classification_report(preds_nb, y_test))
print(f"время: {execution_time} секунд")


# Сохранение веткоризатора, классификатора, базы стоп-слов:
with open('models/task_vectorizer.pkl', 'wb') as vectorizer_file:
    cloudpickle.dump(tfidf_vectorizer, vectorizer_file)

with open('models/task_classifier.pkl', 'wb') as classifier_file:
    pickle.dump(lsvc_clf, classifier_file)

with open('stopwords/russian_stopwords.pkl', 'wb') as stopwords:
    pickle.dump(nltk.corpus.stopwords.words("russian"), stopwords)

with open('models/nltk_punctuation.pkl', 'wb') as punctuation:
    pickle.dump(nltk.corpus.punctuation, punctuation)

print("Обученные модели сохранены в файлы")
