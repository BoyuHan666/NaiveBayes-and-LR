from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

text_clf_nb = Pipeline([
        # ('vec', CountVectorizer(stop_words="english")),
        # ('vec', HashingVectorizer(stop_words="english",non_negative=True)),
        # ('tf_idf', TfidfTransformer()),
        # ('tf_idf', OneHotEncoder()),
        ('tf_idf', TfidfVectorizer(stop_words="english")),
        ('clf', MultinomialNB(alpha=0.05)),
])

text_clf_nb2 = Pipeline([
        ('vec', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('clf', BernoulliNB()),
])

text_clf_svm = Pipeline([
    ('vect', CountVectorizer(stop_words="english")),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                         max_iter=5, tol=None)),
])

# type for X_tain, X_test is numpy and for Y_train, Y_test is list
def getdata1():
    data1_train_set = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data1_test_set = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    X_train = data1_train_set.data
    Y_train = data1_train_set.target
    X_test = data1_test_set.data
    Y_test = data1_test_set.target
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = getdata1()
    # y = []
    # l = []
    # for i in range(20):
    #     c = np.count_nonzero(Y_train == i)
    #     y.append(c)
    #     l.append(i)
    # plt.title('Data1 class distribution')
    # plt.pie(y, labels=l)
    # plt.show()
    # plt.title('Data1 class distribution')
    # l.append(20)
    # plt.hist(Y_train,bins=l)
    # plt.xlim([0,20])
    # plt.xticks(np.arange(0, 21, 1))
    # plt.show()

    # print(len(data1_train_set.data))
    # print(data1_train_set.target[:10])

    cv = HashingVectorizer(stop_words='english')
    X_train_counts = cv.fit_transform(X_train)
    # print(X_train_counts.shape)
    # print(cv.vocabulary_.get(u'happy'))

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # X_train_counts = cv.fit_transform(X_train)
    # vocabulary = []
    # for word in cv.vocabulary_.keys():
    #     try:
    #         int("ncn")
    #         # if int(word):
    #         #    continue
    #     except:
    #         if len(word) > 0 and len(word) < 1000:
    #             vocabulary.append(word)
    # print("num of features: " + str(len(vocabulary)))
    # cv2 = CountVectorizer(stop_words='english', vocabulary=vocabulary)
    cv2 = HashingVectorizer(stop_words='english')
    X_test_counts = cv2.fit_transform(X_test)
    # tfidf_transformer = TfidfTransformer()
    # X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

    clf = MultinomialNB().fit(abs(X_train_counts), Y_train)
    y_bar = clf.predict(abs(X_test_counts))
    print(np.mean(y_bar == Y_test))
    # docs_new = ['God is love', 'OpenGL on the GPU is fast']
    # X_new_counts = cv.transform(docs_new)
    # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    # predicted = clf.predict(X_new_tfidf)

    # use multi naive bayes
    # model1 = text_clf_nb.fit(X_train, Y_train)
    #
    # y_bar = model1.predict(X_test)
    # print(np.mean(y_bar == Y_test))

    # use svm
    # text_clf_svm.fit(X_train, Y_train)
    # y_bar = text_clf_svm.predict(X_test)
    # print(np.mean(y_bar == Y_test))

    # use bernoulli naive bayes
    # text_clf_nb2.fit(X_train, Y_train)
    # y_bar = text_clf_nb2.predict(X_test)
    # print(np.mean(y_bar == Y_test))

    # use gaussian naive bayes
    # cv = CountVectorizer()
    # X_train_counts = cv.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).todense()
    #
    # cv2 = CountVectorizer()
    # X_test_counts = cv2.fit_transform(X_test)
    # tfidf_transformer = TfidfTransformer()
    # X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts).todense()
    # print(X_test_tfidf.shape)
    # print(X_train_tfidf.shape)
    # clf = GaussianNB().fit(X_train_tfidf, Y_train)
    # y_bar = clf.predict(X_test_tfidf)
    # print(np.mean(y_bar == Y_test))


