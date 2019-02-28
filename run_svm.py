'''
Machine Learning classification for character and word ngram

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


'''

Feature Selection with Information Gain

Author: Can Liu
Modification: Ken Steimel

If you want to use this method, please contact them and cite at least one of the following papers:
Kübler, Sandra, Can Liu, and Zeeshan Ali Sayyed. "To use or not to use: Feature selection for sentiment analysis of highly imbalanced data." Natural Language Engineering 24.1 (2018): 3-37
Liu, Can, Sandra Kübler, and Ning Yu. "Feature selection for highly skewed sentiment analysis tasks." Proceedings of the Second Workshop on Natural Language Processing for Social Media (SocialNLP). 2014.

'''

class MultiIG:

    def __init__(self, matrix, labels, vocab):
        self.ig_dict = {}
        self.indexes_dict = {}
        self.chosen_features = []
        self.matrix = matrix
        self.labels = labels
        self.vocab = vocab
        if len(labels) != self.matrix.shape[0]:
            print("Lengths of labels and matrix do not match")

    def perform_feature_selection(self, num_features, refresh=False):
        if refresh or not self.ig_dict:
            self.create_mutual_info_dicts()
            # self.create_dicts()

        self.choose_top_features(num_features)
        return self.filter_matrix()

    def choose_top_features(self, num_features):
        ig_list = [[v, k] for k, v in self.ig_dict.items()]
        ig_list.sort()
        # with open("selected_features" + str(num_features) + ".txt", "w",encoding='utf-8') as out_fp:
        #     for piece in ig_list:
        #         out_fp.write(str(piece[0]) + "\t" + str(piece[1]) + "\n")

        cut_off = len(ig_list) - num_features
        cut_off = max(0, cut_off)
        self.chosen_features = ig_list[cut_off:]
        # print "Done selecting features! The number of features that Information Gain used:", len(ig_list[cut_off:])
        # print "The name of this feature file is called {0} : Make sure to use a different FOLD NAME for different data partitions.".format(CORPATH + fold_name +"_IG_{0}.features".format(this_f) )

        # pickle.dump(ig_dict,open(CORPATH + fold_name + "_igdict.pickle","wb"))

    def filter_matrix(self, matrix=None):
        """
        Author: Ken Steimel
        Date: 2018 10 08
        This method will create a boolean mask from the chosen features list

        """
        print("Filtering matrix...")
        indexes = [self.indexes_dict[feature[1]] for feature in self.chosen_features]
        indexes.sort()
        if matrix != None:
            return matrix[:, indexes]
        else:
            return self.matrix[:, indexes]

    def create_mutual_info_dicts(self):
        """
        This is a version to compare against the results from using create_dicts with my
        self-coded version of mutual information
        """
        res = dict(zip(self.vocab, mutual_info_classif(self.matrix, self.labels)))
        self.ig_dict = res
        self.indexes_dict = dict(zip(self.vocab, range(len(self.vocab))))

    def create_dicts(self):
        """
        Instead of having one dictionary holding feature names and information gain values
        This method produces two dictionaries, one that has the feature names and information gain values
        and another that has features as keys and indexes for the columnn as vals
        """
        print("There are " + str(self.matrix.shape[1]) + " features and ")
        print(str(self.matrix.shape[0]) + " instances to consider")
        possible_labels = list(set(self.labels))
        matricies = {}
        ig_dict = {}
        indexes_dict = {}
        sums = {}
        probabilities = {}
        total_sum = float(self.matrix.sum())
        ig_term1 = 0
        for label in possible_labels:
            row_slice = [True if val == label else False for val in self.labels]
            matricies[label] = self.matrix[row_slice, :]
            sums[label] = float(matricies[label].sum())
            probabilities[label] = max(sums[label] / total_sum, 0.00000000001)
            ig_term1 += probabilities[label] * log(probabilities[label])

        ig_term1 *= -1
        print("Calculating information gain for feature: ")
        print("\r0", end='')
        for col_index in range(len(self.vocab)):
            if col_index % 100 == 0:
                print("\r" + str(col_index), end="")
            term = self.vocab[col_index]
            t_count = max(float(self.matrix[:, col_index].sum()), 0.00000000001)
            label_counts = {}
            ig_term2 = 0
            ig_term3 = 0
            p_t = float(t_count) / total_sum
            p_tbar = 1 - p_t
            for label in possible_labels:
                try:
                    label_counts[label] = float(a_matrix[:, col_index].sum())
                except:
                    label_counts[label] = 0.0
                    p_c1_t = max(label_counts[label] / t_count, 0.00000000001)
                    ig_term2 += p_c1_t * log(p_c1_t)
                    p_c1_tbar = max((sums[label] - label_counts[label]) / (total_sum - t_count), 0.00000000001)
                    ig_term3 += p_c1_tbar * log(p_c1_tbar)

            ig_term2 *= p_t
            ig_term3 *= p_tbar
            ig = ig_term1 + ig_term2 + ig_term3
            # print ig
            ig_dict[term] = ig
            indexes_dict[term] = col_index

        self.ig_dict = ig_dict
        self.indexes_dict = indexes_dict


def data_process():
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

    train_bool_matrix = vectorizer_word.fit_transform(x_train)
    test_bool_matrix = vectorizer_word.transform(x_test)

    lb = LabelEncoder()
    training_data_Y = lb.fit_transform(y_train)

    # scipy.io.mmwrite('../training-data-svm/train_matrix_bool.mtx', train_bool_matrix)
    # scipy.io.mmwrite('../trial-data-svm/trial_matrix_bool.mtx', test_bool_matrix)

    ### Feature Selection ###
    ig_obj = MultiIG(train_bool_matrix, training_data_Y, list(vectorizer_char.vocabulary_.keys()))
    for f_length in 1000,2000,5000:
        training_res = ig_obj.perform_feature_selection(f_length)
        test_res = ig_obj.filter_matrix(test_bool_matrix)
        scipy.io.mmwrite('./training-data-svm/train_simp_matrix_char' + str(f_length) + '.mtx', training_res)
        scipy.io.mmwrite('./trial-data-svm/trial_simp_matrix_char' + str(f_length) + '.mtx', test_res)


def classify():
    train_file_path = open("./training-data/offenseval-training-taskb.tsv")
    test_file_path = open("./trial-data/offenseval-trial-taskb.tsv")

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





