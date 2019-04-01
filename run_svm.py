'''
Machine Learning classification for character ngram

Zack

'''


import os
import scipy.sparse
import scipy.io
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from math import log
from scipy import io
from scipy import sparse
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import BernoulliNB


class MultiIG:

    def __init__(self, matrix, labels, vocab):
        self.ig_dict = {}
        self.indexes_dict = {}
        self.chosen_features = []
        self.matrix = matrix
        self.labels = labels
        self.vocab = vocab
        print("for full code, please contact the author")
'''
Feature Selection with Information Gain
Author: Can Liu
Modification: Ken Steimel
If you want to use this method, please contact them and cite at least one of the following papers:
Kübler, Sandra, Can Liu, and Zeeshan Ali Sayyed. "To use or not to use: Feature selection for sentiment analysis of highly imbalanced data." Natural Language Engineering 24.1 (2018): 3-37
Liu, Can, Sandra Kübler, and Ning Yu. "Feature selection for highly skewed sentiment analysis tasks." Proceedings of the Second Workshop on Natural Language Processing for Social Media (SocialNLP). 2014.

'''


def data_process():
    #Sub-task C as an example, Sub-task b is quite similar with it.
    train_file_path = r'./training-data/offenseval-training-taskb-tweet.tsv'
    test_file_path = r'./trial-data/offenseval-trial-taskb-tweet.tsv'
    train_set = pd.read_csv(train_file_path, sep="\t",header=None)
    x_train = train_set.iloc[:, 0].values
    y_train = train_set.iloc[:, 1].values

    test_set = pd.read_csv(test_file_path, sep="\t",header=None)
    x_test = test_set.iloc[:, 0].values

    vectorizer_char = CountVectorizer(strip_accents="unicode",
                                 analyzer="char",
                                 ngram_range=(1, 7),
                                 min_df=1,
                                 binary=True)

    train_bool_matrix = vectorizer_char.fit_transform(x_train)
    test_bool_matrix = vectorizer_char.transform(x_test)

    lb = LabelEncoder()
    training_data_Y = lb.fit_transform(y_train)

    ### Feature Selection ###
    ig_obj = MultiIG(train_bool_matrix, training_data_Y, list(vectorizer_char.vocabulary_.keys()))
    for f_length in 1000,2000,5000:
        training_res = ig_obj.perform_feature_selection(f_length)
        test_res = ig_obj.filter_matrix(test_bool_matrix)
        scipy.io.mmwrite('./training-data-svm/train_matrix_char' + str(f_length) + '.mtx', training_res)
        scipy.io.mmwrite('./trial-data-svm/trial_matrix_char' + str(f_length) + '.mtx', test_res)

def classify():
    train_file_path = open("./training-data/offenseval-training-taskb.tsv")
    test_file_path = open("./trial-data/offenseval-trial-taskb.tsv")

    # add linguistic, emoji and entity features
    trainingData_emoji = open("./training-data/training-taskb-emoji.tsv")
    testingData_emoji= open("./trial-data/trial-taskb-emoji.tsv")

    train_set = pd.read_csv(train_file_path, sep="\t",header=None)
    y_train = train_set.iloc[:, 1].values

    test_set = pd.read_csv(test_file_path, sep="\t",header=None)
    y_test = test_set.iloc[:, 1].values

    lb = LabelEncoder()
    trainingDataY = lb.fit_transform(y_train)
    testingDataY = lb.transform(y_test)

    trainingDataXpre = "train_matrix_bool.mtx"
    testingDataXpre = "trial_matrix_bool.mtx"

    trainingDataXnp1 = io.mmread(trainingDataXpre)
    trainingDataXnp2 = np.genfromtxt(trainingData_emoji,delimiter='\t')

    testingDataXnp1 = io.mmread(testingDataXpre)
    testingDataXnp2 = np.genfromtxt(testingData_emoji,delimiter='\t')


    # Need to change y data into single iterable without
    # iterables inside it
    trainingDataY = np.ravel(trainingDataY)
    testingDataY = np.ravel(testingDataY)

    # Covert data into NumPy arrays

    trainingDataYnp = np.array(trainingDataY)
    testingDataYnp = np.array(testingDataY)

    trainingDataXnp1 = sparse.csr_matrix(trainingDataXnp1)
    testingDataXnp1 = sparse.csr_matrix(testingDataXnp1)
    trainingDataXnp1= trainingDataXnp1.toarray()
    testingDataXnp1 = testingDataXnp1.toarray()

    nm=Normalizer()
    trainingDataXnp=np.hstack((trainingDataXnp1, trainingDataXnp2))
    trainingDataXnp = nm.fit_transform(trainingDataXnp)

    testingDataXnp = np.hstack((testingDataXnp1,testingDataXnp2))
    testingDataXnp = nm.transform(testingDataXnp)

    print(trainingDataXnp.shape)
    print(testingDataXnp.shape)

    # SVM
    svc = svm.LinearSVC()
    pipeline = svc
    param_grid = {'loss': ['hinge', 'squared_hinge'],
                  'random_state': [5302],
                  'C': [0.01, 0.1, 1, 10]}
    model = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=1, cv=8)

    model.fit(trainingDataXnp, trainingDataYnp)
    predictingY = model.predict(testingDataXnp)

    #save predict labels for further examining
    # np.savetxt("predict.tsv", predictingY,  delimiter="\t")

    # Print the results
    print("Model accuracy: ", accuracy_score(testingDataYnp, predictingY))
    print ("F1 score: ",f1_score(testingDataYnp,predictingY,average='macro'))
    print(classification_report(testingDataYnp, predictingY, digits=8))
    print(model.best_params_)


if __name__ == '__main__':

    # data_process()
    classify()





