import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
    def __init__(self, threshold):
        self.class_prob = None
        self.class_features_prob = None
        self.class_list = None
        self.num_of_vocabulary = None
        self.threshold = threshold
        return

    def fit(self, alpha, X_train, Y_train):
        num_instances = len(X_train)
        num_features = len(X_train[0])
        self.num_of_vocabulary = num_features
        self.class_list = list(set(Y_train))
        num_class = len(self.class_list)
        self.class_prob = np.zeros(num_class)
        self.class_features_prob = np.zeros((num_class, num_features))

        for cls in range(num_class):
            X_train_cls = X_train[Y_train == self.class_list[cls]]
            all_words_in_cls = np.sum(X_train_cls)
            self.class_prob[cls] = len(X_train_cls) / num_instances
            for f in range(num_features):
                f_counts = np.sum(X_train_cls[:, f])
                self.class_features_prob[cls, f] = (f_counts + alpha) / (all_words_in_cls + alpha * num_features)
        return self

    def predict(self, X_test):
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
            self.class_prob[cls] = (len(X_train_cls) + smooth) / (num_instances + smooth * num_class)
            self.mean[cls, :] = np.mean(X_train_cls, 0) + self.threshold
            self.sigma[cls, :] = np.std(X_train_cls, 0) + self.threshold
        return self

    def predict(self, X_test):
        likelihoods = -0.5 * np.log(2 * np.pi) - np.log(self.sigma[:, None, :]) - (
                (X_test[None, :, :] - self.mean[:, None, :]) ** 2) / (2 * self.sigma[:, None, :] ** 2)
        likelihoods = np.sum(likelihoods, axis=2)
        posterior = np.log(self.class_prob)[:, None] + likelihoods
        predictions = []
        for llh in posterior.T:
            class_idx = np.argmax(llh)
            predictions.append(self.class_list[class_idx])
        return predictions


class KFoldCrossValidation:
    def __init__(self):
        self.k = None
        return

    def fit(self, k):
        self.k = k
        return self

    def predict(self, X_text):
        return

    def cross_validation_split(self, seed, percentage, X_train, Y_train):
        # np.random.seed(seed)
        kfold = self.k
        ins_num = len(Y_train)
        Y_trainT = np.reshape(Y_train, (ins_num, 1))
        data = np.concatenate((X_train, Y_trainT), axis=1)
        np.random.shuffle(data)
        data = data[0:int(ins_num * percentage)]
        ins_num = len(data)
        one_fold_ins_num = ins_num // kfold
        start = 0
        fold_list = []
        train_list = []
        for i in range(kfold):
            end = start + one_fold_ins_num
            fold_list.append(data[start:end])
            train_list.append(np.concatenate((data[0:start], data[end:-1]), axis=0))
            start += one_fold_ins_num
        return fold_list, train_list

    def kfoldCV(self, fold_list, train_list, model, smooth, threshold):
        cum_acc = 0
        num = len(fold_list)
        for i in range(num):
            validation_set = fold_list[i]
            validation_x = validation_set[:, 0:-1]
            validation_y = validation_set[:, -1]

            train_set = train_list[i]
            train_x = train_set[:, 0:-1]
            train_y = train_set[:, -1]

            if (model == MultinomialNaiveBayes) or (model == GaussianNaiveBayes):
                mdl = model(threshold)
                mdl.fit(smooth, train_x, train_y)
            else:
                mdl = model(penalty='l2',C=0.61,max_iter=60)
                # mdl = model()
                mdl.fit(train_x, train_y)
            y_bar = mdl.predict(validation_x)
            acc = evaluate_acc(y_bar, validation_y.tolist())
            # print("the accuracy is: " + "{:.2f}".format(acc) + "%")
            cum_acc += acc
        return cum_acc / num


def generate_result_data1(model, max_features, k, percentage_list, smooth, threshold):
    # test acc for data 1
    print("----------testing data1----------")
    X_train, Y_train, _, _ = getdata1()

    cv = CountVectorizer(max_features=max_features, stop_words='english')
    X_train_counts = cv.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfc = KFoldCrossValidation()
    kfc.fit(k=k)
    avg_accs = []
    for percentage in percentage_list:
        f, t = kfc.cross_validation_split(seed=1, percentage=percentage, X_train=X_train_tfidf.toarray(),
                                          Y_train=Y_train)
        acc = kfc.kfoldCV(fold_list=f, train_list=t, model=model, smooth=smooth, threshold=threshold)
        print("the average accuracy of " + str(percentage * 100) + "% data is: " + "{:.2f}".format(acc) + "%")
        avg_accs.append(acc)
    return avg_accs


def generate_result_data2(instance_range, model, max_features, k, percentage_list, smooth, threshold):
    # test acc for data 2
    print("----------testing data2----------")
    X_train, Y_train, _, _ = getdata2()
    X_train = np.concatenate((X_train[0:instance_range], X_train[-instance_range:-1]), axis=0)
    Y_train = np.concatenate((Y_train[0:instance_range], Y_train[-instance_range:-1]), axis=0)

    cv = CountVectorizer(max_features=max_features, stop_words='english')
    X_train_counts = cv.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfc = KFoldCrossValidation()
    kfc.fit(k=k)
    avg_accs = []
    for percentage in percentage_list:
        f, t = kfc.cross_validation_split(seed=1, percentage=percentage, X_train=X_train_tfidf.toarray(),
                                          Y_train=Y_train)
        acc = kfc.kfoldCV(fold_list=f, train_list=t, model=model, smooth=smooth, threshold=threshold)
        print("the average accuracy of " + str(percentage * 100) + "% data is: " + "{:.2f}".format(acc) + "%")
        avg_accs.append(acc)
    return avg_accs


def plot(k, x, ys, labels):
    plt.title(str(k)+'-fold')
    for i in range(len(ys)):
        plt.plot(x, ys[i], '-o', label=labels[i])
        for j in range(len(x)):
            plt.text(x[j], ys[i][j], s="{:.2f}".format(ys[i][j]))
    plt.xlabel('percentage of data')
    plt.ylabel('average accuracy (%)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    percentage_list = [0.2, 0.4, 0.6, 0.8, 1]
    np.random.seed(123)
    # percentage_list = [1]
    k=5
    max_features = 10000  # how many features(words) for training
    instance_range = 20000  # how many data we want to use as the whole train set
    # plot for data1
    # np.random.seed(1)
    # avg_accs_NB1 = generate_result_data1(model=MultinomialNaiveBayes, max_features=max_features, k=k,
    #                                      percentage_list=percentage_list, smooth=0.05, threshold=1e-5)
    # avg_accs_Log1 = generate_result_data1(model=LogisticRegression, max_features=max_features, k=k,
    #                                       percentage_list=percentage_list, smooth=0.05, threshold=1e-5)
    # plot(k, percentage_list,[avg_accs_NB1,avg_accs_Log1],["MNB","LogR"])

    # plot for data2
    # avg_accs_NB2 = generate_result_data2(instance_range=instance_range//2, model=MultinomialNaiveBayes,
    #                                      max_features=max_features, k=k, percentage_list=percentage_list, smooth=0.01,
    #                                      threshold=1e-5)
    # avg_accs_Log2 = generate_result_data2(instance_range=instance_range//2, model=LogisticRegression,
    #                                       max_features=max_features, k=k, percentage_list=percentage_list, smooth=0.1,
    #                                       threshold=1e-5)
    # plot(k, percentage_list, [avg_accs_NB2, avg_accs_Log2], ["MNB", "LogR"])
