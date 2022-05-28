import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

text_clf_nb = Pipeline([
        # ('vec', CountVectorizer(stop_words='english')),
        ('vec', TfidfVectorizer(stop_words='english')),
        # ('tf_idf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.01)),
])

text_clf_nb2 = Pipeline([
        ('vec', CountVectorizer(stop_words='english')),
        ('tf_idf', TfidfTransformer()),
        ('clf', BernoulliNB()),
])

text_clf_svm = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', random_state=42,
                         alpha=1e-3, max_iter=5, tol=None)),
])

# type for X_tain, X_test is numpy and for Y_train, Y_test is list
def getdata2():
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df_train = pd.read_csv("./Data/training.1600000.processed.noemoticon.csv", header=None, names=cols,
                           encoding='latin-1')
    df_test = pd.read_csv("./Data/new_testdata.manual.2009.06.14.csv", header=None, names=cols, encoding='latin-1')

    data2_train = df_train.to_numpy()
    X_train = data2_train[:, 5]
    Y_train = data2_train[:, 0].tolist()

    data2_test = df_test.to_numpy()
    X_test = data2_test[:, 5]
    Y_test = data2_test[:, 0].tolist()
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getdata2()
    # y = [np.count_nonzero(np.array(Y_train) == 0),np.count_nonzero(np.array(Y_train) == 4)]
    # l = [0,4]
    # plt.title('Data2 class distribution')
    # plt.pie(y, labels=l)
    # plt.show()
    # plt.title('Data2 class distribution')
    # plt.hist(Y_train)
    # plt.show()

    cv = HashingVectorizer(stop_words='english')
    X_train_counts = cv.fit_transform(X_train)
    X_test_counts = cv.fit_transform(X_test)
    clf = MultinomialNB().fit(abs(X_train_counts), Y_train)
    y_bar2 = clf.predict(abs(X_test_counts))
    print(np.mean(y_bar2 == Y_test))

    # use multinomial naive bayes
    # model2 = text_clf_nb.fit(X_train, Y_train)
    # y_bar2 = model2.predict(X_test)
    # print(np.mean(y_bar2 == Y_test))
    #
    # # use svm
    # model2 = text_clf_svm.fit(X_train, Y_train)
    # y_bar2 = model2.predict(X_test)
    # print(np.mean(y_bar2 == Y_test))

    # use bernoulli naive bayes
    # model2 = text_clf_nb2.fit(X_train, Y_train)
    # y_bar = model2.predict(X_test)
    # print(np.mean(y_bar == Y_test))

    # # use gaussian naive bayes
    # cv = CountVectorizer()
    # X_train_counts = cv.fit_transform(X_train)
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #
    # print(type(X_train_tfidf))
    # print(type(X_train))
    # # clf = GaussianNB().fit(X_train_tfidf, X_train)
