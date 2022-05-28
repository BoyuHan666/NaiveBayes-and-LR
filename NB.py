import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

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
    df_train = pd.read_csv("../../Task1/Data/training.1600000.processed.noemoticon.csv", header=None, names=cols,
                           encoding='latin-1')
    df_test = pd.read_csv("../../Task1/Data/new_testdata.manual.2009.06.14.csv", header=None, names=cols,
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
        if l1[i] == l2[i]:
            count += 1
    acc = count / len(l2) * 100
    return acc


class MultinomialNaiveBayes:
    # introduce threshold to avoid RuntimeWarning for log
    def __init__(self,threshold):
        self.class_prob = None
        self.class_features_prob = None
        self.class_list = None
        self.num_of_vocabulary = None
        self.threshold = threshold
        return

    def fit(self, alpha, X_train, Y_train):
        print("fitting")
        num_instances = len(X_train)
        num_features = len(X_train[0])
        self.num_of_vocabulary = num_features
        self.class_list = list(set(Y_train))
        num_class = len(self.class_list)
        self.class_prob = np.zeros(num_class)
        self.class_features_prob = np.zeros((num_class, num_features))
        Y_train = np.array(Y_train)
        for cls in range(num_class):
            X_train_cls = X_train[Y_train == self.class_list[cls]]
            all_words_in_cls = np.sum(X_train_cls)
            self.class_prob[cls] = len(X_train_cls) / num_instances + self.threshold
            for f in range(num_features):
                f_counts = np.sum(X_train_cls[:, f])
                self.class_features_prob[cls, f] = (f_counts + alpha) / (all_words_in_cls + alpha * num_features)
        return self

    def predict(self, X_test):
        print("predicting")
        likelihoods = np.matmul(X_test, np.log(self.class_features_prob + self.threshold).T)
        likelihoods = likelihoods + np.log(self.class_prob + self.threshold)
        predictions = []
        for llh in likelihoods:
            class_idx = np.argmax(llh)
            predictions.append(self.class_list[class_idx])
        return predictions

class GaussianNaiveBayes():
    def __init__(self, threshold):
        self.mean = None
        self.sigma = None
        self.class_prob = None
        self.class_list = None
        self.num_of_vocabulary = None
        self.threshold = threshold
        return

    def fit(self, smooth, X_train, Y_train):
        print("fitting")
        num_instances = len(X_train)
        num_features = len(X_train[0])
        class_list = list(set(Y_train))
        self.class_list = class_list
        num_class = len(self.class_list)
        self.mean = np.zeros((num_class, num_features))
        self.sigma = np.zeros((num_class, num_features))
        Y_train = np.array(Y_train)
        self.class_prob = np.zeros(num_class)
        for cls in range(num_class):
            X_train_cls = X_train[Y_train == self.class_list[cls]]
            self.class_prob[cls] = (len(X_train_cls)+smooth) / (num_instances+smooth*num_class)
            self.mean[cls, :] = np.mean(X_train_cls, 0) + self.threshold
            self.sigma[cls, :] = np.std(X_train_cls, 0) + self.threshold
        return self

    def predict(self, X_test):
        print("predicting")
        likelihoods = -0.5 * np.log(2 * np.pi) - np.log(self.sigma[:, None, :]) - ((X_test[None, :, :] - self.mean[:, None, :])**2) / (2*self.sigma[:, None, :]**2)
        likelihoods = np.sum(likelihoods, axis=2)
        posterior = np.log(self.class_prob)[:, None] + likelihoods
        predictions = []
        for llh in posterior.T:
            class_idx = np.argmax(llh)
            predictions.append(self.class_list[class_idx])
        return predictions


if __name__ == '__main__':

    # test acc for data 1
    # print("----------testing data1----------")
    # X_train, Y_train, X_test, Y_test = getdata1()
    # X_test = X_test[:1000]
    # Y_test = Y_test[:1000]
    # cv = CountVectorizer(max_features=10000, stop_words='english')
    # # cv = CountVectorizer(max_features=120000)
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
    # # cv2 = CountVectorizer(stop_words='english', vocabulary=vocabulary)
    # cv2 = CountVectorizer(vocabulary=vocabulary)
    # X_train_counts = cv2.fit_transform(X_train)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # # generate test tf-idf
    # X_test_counts = cv2.fit_transform(X_test)
    # tfidf_transformer = TfidfTransformer()
    # X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
    #
    # nb = MultinomialNaiveBayes(0)
    # alpha = 0.05
    # best_acc = 0
    # while alpha<0.1:
    #     nb.fit(alpha, X_train_tfidf.toarray(), Y_train)
    #
    #     y_bar = nb.predict(X_test_tfidf.toarray())
    #     acc = evaluate_acc(y_bar, Y_test)
    #     print("#######")
    #     print("alpha = "+str(alpha))
    #     print("the accuracy of MultinomialNaiveBayes is: " + "{:.2f}".format(acc) + "%")
    #     if acc < best_acc:
    #         break
    #     best_acc = acc
    #     alpha += 0.05
    #
    # gnb = GaussianNaiveBayes(0.01)
    # # gnb = BernoulliNB()
    # gnb.fit(1, X_train_tfidf.toarray(), Y_train)
    # # gnb.fit(X_train_tfidf.toarray(), Y_train)
    # gy_bar = gnb.predict(X_test_tfidf.toarray())
    # gacc = evaluate_acc(gy_bar, Y_test)
    # print("#######")
    # print("the accuracy of GaussianNaiveBayes is: " + "{:.2f}".format(gacc) + "%")

    # test acc for data 2
    print("----------testing data2----------")
    X_train, Y_train, X_test, Y_test = getdata2()
    X_test = X_test[:1000]
    Y_test = Y_test[:1000]
    X_train = np.concatenate((X_train[0:5000], X_train[-5000:-1]), axis=0)
    Y_train = np.concatenate((Y_train[0:5000], Y_train[-5000:-1]), axis=0)
    cv = CountVectorizer(max_features=10000, stop_words='english')
    X_train_counts = cv.fit_transform(X_train)
    vocabulary = []
    for word in cv.vocabulary_.keys():
        try:
            if int(word):
                continue
        except:
            vocabulary.append(word)
    print("num of features: " + str(len(vocabulary)))
    cv2 = CountVectorizer(stop_words='english', vocabulary=vocabulary)
    X_train_counts = cv2.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # generate test tf-idf
    X_test_counts = cv2.fit_transform(X_test)
    tfidf_transformer = TfidfTransformer()
    X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

    nb = MultinomialNaiveBayes(0)
    nb.fit(0.1, X_train_tfidf.toarray(), Y_train)

    y_bar = nb.predict(X_test_tfidf.toarray())
    acc = evaluate_acc(y_bar, Y_test)
    print("#######")
    print("the accuracy of MultinomialNaiveBayes is: " + "{:.2f}".format(acc) + "%")

    gnb = GaussianNaiveBayes(0.01)
    gnb.fit(1, X_train_tfidf.toarray(), Y_train)
    gy_bar = gnb.predict(X_test_tfidf.toarray())
    gacc = evaluate_acc(gy_bar, Y_test)
    print("#######")
    print("the accuracy of GaussianNaiveBayes is: " + "{:.2f}".format(gacc) + "%")