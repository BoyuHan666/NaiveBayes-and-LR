from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

text_clf_lr = Pipeline([
    ('vec', CountVectorizer(max_features=10000, stop_words="english")),
    # ('vec', CountVectorizer(max_features=500000)),
    ('tf_idf', TfidfTransformer()),
    ('clf', LogisticRegression(penalty='l2',C=0.61,max_iter=60)),
])

text_clf_li = Pipeline([
    ('vec', CountVectorizer(max_features=1000, stop_words="english")),
    ('tf_idf', TfidfTransformer()),
    ('clf', LinearRegression(fit_intercept=False)),
])


def getdata1():
    data1_train_set = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data1_test_set = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    X_train = data1_train_set.data
    Y_train = data1_train_set.target
    X_test = data1_test_set.data
    Y_test = data1_test_set.target
    return X_train, Y_train, X_test, Y_test


def getdata2():
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df_train = pd.read_csv("../Task1/Data/training.1600000.processed.noemoticon.csv", header=None, names=cols,
                           encoding='latin-1')
    df_test = pd.read_csv("../Task1/Data/new_testdata.manual.2009.06.14.csv", header=None, names=cols,
                          encoding='latin-1')

    data2_train = df_train.to_numpy()
    X_train = data2_train[:, 5]
    Y_train = data2_train[:, 0].tolist()

    data2_test = df_test.to_numpy()
    X_test = data2_test[:, 5]
    Y_test = data2_test[:, 0].tolist()
    return X_train, Y_train, X_test, Y_test

def evaluate_acc(l1, l2):
    count = 0
    for i in range(len(l1)):
        if int(l1[i]) == int(l2[i]):
            count += 1
    acc = count / len(l2) * 100
    return acc

if __name__ == '__main__':
    # test acc of data1
    print("----------testing data1----------")
    X_train1, Y_train1, X_test1, Y_test1 = getdata1()
    # model1 = text_clf_lr.fit(X_train1, Y_train1)
    # y_bar = model1.predict(X_test1)
    # acc = evaluate_acc(y_bar, Y_test1)
    # print("the accuracy is: " + "{:.2f}".format(acc) + "%")

    linear1 = text_clf_li.fit(X_train1, Y_train1)
    y_bar = linear1.predict(X_test1)
    acc = evaluate_acc(y_bar, Y_test1)
    print("the accuracy is: " + "{:.2f}".format(acc) + "%")

    # test acc of data2
    print("----------testing data2----------")
    X_train2, Y_train2, X_test2, Y_test2 = getdata2()
    # CPU is not enough to run all the train set
    X_train2 = np.concatenate((X_train2[0:50000], X_train2[-50000:-1]), axis=0)
    Y_train2 = np.concatenate((Y_train2[0:50000], Y_train2[-50000:-1]), axis=0)
    # model2 = text_clf_lr.fit(X_train2, Y_train2)
    # y_bar = model2.predict(X_test2)
    # acc2 = evaluate_acc(y_bar, Y_test2)
    # print("the accuracy is: " + "{:.2f}".format(acc2) + "%")
    #
    linear2 = text_clf_li.fit(X_train2, Y_train2)
    y_bar = linear2.predict(X_test2)
    acc2 = evaluate_acc(y_bar, Y_test2)
    print("the accuracy is: " + "{:.2f}".format(acc2) + "%")
