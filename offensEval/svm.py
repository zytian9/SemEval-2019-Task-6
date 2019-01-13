#!/usr/bin/python

from __future__ import print_function
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm.libsvm import predict_proba
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.svm import LinearSVC
from scipy import sparse
import numpy as np
import pickle as pkl
from scipy import io
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import sys

def readFile(fileReader):
    """

    Function for reading in files and splitting the columns into
    individual list elements.

    e.g.

    `[[cell1,line1],[cell2,line1]...`

    However, note that the training data is already quantitative.
    This result is used directly by sklearn in this application
    """
    lines = fileReader.readlines()
    listMatrix = []
    for line in lines:
        linelist = line.strip().split('\t')
        linelist = [float(x) for x in linelist]
        listMatrix.append(linelist)

    return listMatrix

def main():
    # Read in training data and testing data
    count = ''
    if len(sys.argv) == 2:
        count = sys.argv[1]
        print(count)

    trainingDataYpre = open("./offenseval-training-task-a.tsv")
    testingDataYpre = open("./offenseval-trial-task-a.tsv")

    trainingDataXpre = ''
    testingDataXpre = ''

    if count == '':
        trainingDataXpre = "train_a_matrix.mtx"
        testingDataXpre = "test_a_matrix.mtx"

    else:
        # These versions are for boolean features without considering feature selection
        trainingDataXpre = "train_a_matrix_bool.mtx"
        testingDataXpre = "test_a_matrix_bool.mtx"

    trainingDataX = io.mmread(trainingDataXpre)
    trainingDataY = readFile(trainingDataYpre)

    testingDataX = io.mmread(testingDataXpre)
    testingDataY = readFile(testingDataYpre)

    # Need to change y data into single iterable without
    # iterables inside it
    trainingDataY = np.ravel(trainingDataY)
    testingDataY = np.ravel(testingDataY)

    # Covert data into NumPy arrays

    trainingDataYnp = np.array(trainingDataY)
    testingDataYnp = np.array(testingDataY)
    trainingDataXnp = sparse.csr_matrix(trainingDataX)
    testingDataXnp = sparse.csr_matrix(testingDataX)

    print(trainingDataXnp.shape)
    print(testingDataXnp.shape)

    # Run the classifier
    svc = svm.LinearSVC()
    pipeline = svc  # Pipeline([('clf', svc)])
    param_grid = {'loss': ['hinge', 'squared_hinge'],
                  'random_state': [32391],
                  'C': [0.01, 0.1, 1, 10]}

    model = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=1, cv=5)

    model.fit(trainingDataXnp, trainingDataYnp)

    predictingY = model.predict(testingDataXnp)

    # Print the results

    print("Model accuracy: ", accuracy_score(testingDataYnp, predictingY))

    print(classification_report(testingDataYnp, predictingY, digits=8))

    print(model.best_params_)

    with open("outputModel.pkl", "wb") as outputFP:
        pkl.dump(model, outputFP)


if __name__ == '__main__':
    main()