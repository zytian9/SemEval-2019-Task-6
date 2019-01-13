import string
import csv
import os
import scipy.sparse
import scipy.io
import numpy as np
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from math import log


def write_csv(file_path, content, delimiter='\t'):
    with open(file_path, 'w', newline='') as csv_file:
        for row in content:
            row = row.toarray()[0]  # iterators over the rows of a sparse matrix always return a 2d sparse matrix
            text_row = delimiter.join([str(x) for x in row]) + "\n"
            csv_file.write(text_row)
        # writer = csv.writer(csv_file, delimiter=dilimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerows(content)


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


def featurize(corpus, extraction_model='', boolean_feats=False):
    '''
    Tokenizes and creates Count BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of ngram counts.
    '''
    vectorizer = []
    X = []
    if extraction_model:
        X = extraction_model.transform(corpus)
    else:
        vectorizer = CountVectorizer(strip_accents="unicode",
                                     analyzer="char",
                                     ngram_range=(2, 7),
                                     min_df=3,
                                     binary=boolean_feats)
        X = vectorizer.fit_transform(corpus)

    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X, vectorizer

def main():

    train_file_path = r'./offenseval-training-tweet.tsv'
    test_file_path = r'./offenseval-trial-tweet.tsv'
    training_data_Y = readFile(open("./offenseval-training-task-a.tsv"))
    testing_data_Y = readFile(open("./offenseval-trial-task-a.tsv"))
    training_data_Y = np.ravel(training_data_Y)
    testing_data_Y = np.ravel(testing_data_Y)
    # get content from train and test file and combine them in a full_lines
    train_lines = open(train_file_path, 'r', encoding='utf8')
    test_lines = open(test_file_path, 'r', encoding='utf8')
    train_lines = [line for line in train_lines]
    test_lines = [line for line in test_lines]

    train_matrix, extractor = featurize(train_lines)
    test_matrix, dummy = featurize(test_lines, extraction_model=extractor)
    train_bool_matrix, bool_extractor = featurize(train_lines, boolean_feats=True)
    test_bool_matrix, dummy = featurize(test_lines, extraction_model=bool_extractor)
    # output
    sklearn.datasets.dump_svmlight_file(train_matrix,
                                        training_data_Y,
                                        "train_a_matrix.svm")
    sklearn.datasets.dump_svmlight_file(test_matrix,
                                        testing_data_Y,
                                        "test_a_matrix.svm")
    scipy.io.mmwrite('train_a_matrix.mtx', train_matrix)
    scipy.io.mmwrite('test_a_matrix.mtx',test_matrix)

    sklearn.datasets.dump_svmlight_file(train_matrix,
                                        training_data_Y,
                                        "train_a_matrix_bool.svm")
    sklearn.datasets.dump_svmlight_file(test_matrix,
                                        testing_data_Y,
                                        "test_a_matrix_bool.svm")
    scipy.io.mmwrite('train_a_matrix_bool.mtx', train_matrix)
    scipy.io.mmwrite('test_a_matrix_bool.mtx', test_matrix)




if __name__ == '__main__':
    main()





