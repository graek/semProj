############################################################################################
# GROUP MEMBERS:
# Lukas Grässle
# Olga Sozinova
# Xiao'ao Song
# Yue Ding

# We are aware that the proposed solution is not correctly utilizing the extended feature space,
# but we did not manage to implement the CountVecotrizer the right way to combine the Word-count for the BOW
# together with other features (like POS tags).
#
# At the point of the submission, the DictVectorizer produces an error, which we will try to remove until the lecture tomorrow.


############################################################################################

# HOW TO RUN:

# python PA4_part_2.py -tr train.json.txt -ts test.json.txt -f test_labels.txt

import gzip
import numpy as np
import random
import os
import re
import json
import argparse

from collections import Counter, defaultdict, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import numpy as np
import spacy


############################################################################################
# 1. LOAD DATA
############################################################################################

def load_data(file, verbose=True):
    f = open(file, 'r', encoding='utf-8')
    data = []
    labels = []
    for i, line in enumerate(f):
        instance = json.loads(line)
        if i == 0:
            if verbose:
                print('json example:')
                print(instance)
        # 'relation, entity_1, entity_2, snippet' fileds for each example
        # 'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'], instance['entity_2'], [])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'], snippet['mention_1'],
                                        snippet['middle'],
                                        snippet['mention_2'], snippet['right'],
                                        snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i == 0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])
    return data, labels


###########################################################################################
# 2. EXTRACT FEATURES and BUILD CLASSIFIER
###########################################################################################

# Extract BOW
def ExtractBow(data, verbose=False):
    featurized_data = []
    for instance in data:
        bow = set()
        for s in instance.snippet:
            bow.update(s.left.split(), s.middle.split(), s.right.split())
        featurized_data.append(' '.join(bow))
    return featurized_data


# Extract features
def ExtractFeatures(data, read_from_file=False):
    counter = 0
    featurized_data = []
    nlp = spacy.load('en')

    pos_text = []
    syntax_text = []

    if read_from_file:
        f1 = open('pos_combinations.txt', 'r')
        f2 = open('syntax_combinations.txt', 'r')
        pos_text = f1.readlines()
        syntax_text = f2.readlines()

    for instance in data:
        bow = set()
        current_syntax = []
        current_pos = []

        features_dict = {
            'Mention_1': None,
            'Mention_2': None,
            'BOW_between': None,
            'BOW_between_POS': None,
            'BOW_between_syntax': None
        }

        for s in instance.snippet:
            bow.update(s.left.split(), s.middle.split(), s.right.split())

            if len(pos_text) == 0 and len(syntax_text) == 0:
                doc = nlp(s.middle)
                syntax_combination = syntax_features(doc)
                current_syntax.append(syntax_combination)
                pos_combination = pos_tags(doc)
                current_pos.append(pos_combination)

        if len(pos_text) > 0 and len(syntax_text) > 0:
            current_syntax = syntax_text[counter]
            current_pos = pos_text[counter]

        # build a rich feature space for each example
        features_dict['Mention_1'] = instance.entity_1
        features_dict['Mention_2'] = instance.entity_2
        features_dict['BOW_between'] = ' '.join(bow)
        features_dict['BOW_between_POS'] = ' '.join(current_pos)
        features_dict['BOW_between_syntax'] = ' '.join(current_syntax)
        featurized_data.append(features_dict)

        print(counter)
        counter += 1
    return featurized_data


# Syntactic analysis (NP, VP, etc.)
def syntax_features(doc):
    syntax_combination = []
    for token in doc:
        syntax_combination.append(token.dep_)
    return ' '.join(syntax_combination)


# POS tags
def pos_tags(doc):
    pos_combination = []
    for token in doc:
        pos_combination.append(token.pos_)
    return ' '.join(pos_combination)


# Preprocessing: delete punctuation
def delete_punct(line):
    new_line = re.sub(u'[^\w ]+', '', line, flags=re.UNICODE)  # delete punctuation
    new_line = new_line.lower()  # lowercase
    return new_line


# Preprocessing: lemmatize
def lemmatize(doc):
    output_line = []
    for token in doc:
        if not token.is_punct:
            output_line.append(token.lemma_)
    return ' '.join(output_line)


##################################################################################################
# 3. TRAIN CLASSIFIER AND EVALUATE (CV)
##################################################################################################

def print_statistics_header():
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        'relation', 'precision', 'recall', 'f-score', 'support'))
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))


def print_statistics_row(rel, result):
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format(rel, *result))


def print_statistics_footer(avg_result):
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format('macro-average', *avg_result))


def macro_average_results(results):
    avg_result = [np.average([r[i] for r in results.values()]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results.values()]))
    return avg_result


def average_results(results):
    avg_result = [np.average([r[i] for r in results]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results]))
    return avg_result


def evaluateCV(clf, label_encoder, X, y, verbose=True):
    results = {}
    for rel in le.classes_:
        results[rel] = []
    if verbose:
        print_statistics_header()
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_index, test_index in kfold.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            clf.fit(X_train, y_train)
            pred_labels = clf.predict(X_test)
            stats = precision_recall_fscore_support(y_test, pred_labels, beta=0.5)
            # print(stats)
            for rel in label_encoder.classes_:
                rel_id = label_encoder.transform([rel])[0]
                # print(rel_id,rel)
                stats_rel = [stat[rel_id] for stat in stats]
                results[rel].append(stats_rel)
        for rel in label_encoder.classes_:
            results[rel] = average_results(results[rel])
            if verbose:
                print_statistics_row(rel, results[rel])
    avg_result = macro_average_results(results)
    if verbose:
        print_statistics_footer(avg_result)
    return avg_result[2]  # return f_0.5 score as summary statistic


# A check for the average F1 score
f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')


def evaluateCV_check(classifier, X, y, verbose=True):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring=f_scorer)
    print("\nCross-validation scores (StratifiedKFold): ", scores)
    print("Mean cv score (StratifiedKFold): ", scores.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relations")
    parser.add_argument("-tr", "--train", dest="train_path",
                        help="file path to the train set")
    parser.add_argument("-ts", dest="test_path",
                        help="file path to the test set")
    parser.add_argument("-f", dest="result",
                        help="file path to the set T")
    args = parser.parse_args()

    # Load data and labels
    PairExample = namedtuple('PairExample',
                             'entity_1, entity_2, snippet')
    Snippet = namedtuple('Snippet',
                         'left, mention_1, middle, mention_2, right, direction')
    train_data, train_labels = load_data(args.train_path, verbose=False)
    print('Data loaded')

    # MODEL 1

    # Transform dataset to features
    # print('Training model 1...')
    # train_data_featurized_bow = ExtractBow(train_data)
    # print('Model 1 trained')
    # le = LabelEncoder()
    # train_labels_featurized_bow = le.fit_transform(train_labels)
    # bow_clf = make_pipeline(CountVectorizer(), LogisticRegression())
    # bow_clf.fit(train_data_featurized_bow, train_labels_featurized_bow)
    # print('Making predictions for the model 1...')
    # test_data, test_labels = load_data(args.test_path, verbose=False)
    # test_data_featurized_bow = ExtractBow(test_data)
    # test_label_predicted_bow = bow_clf.predict(test_data_featurized_bow)
    # test_label_predicted_decoded_bow = le.inverse_transform(test_label_predicted_bow)
    # print(test_label_predicted_decoded_bow[:2])
    # print('Predictions made')
    # print('CV scores for the model 1:')
    # # Evaluate the model
    # print(evaluateCV(bow_clf, le, train_data_featurized_bow, train_labels_featurized_bow))
    # evaluateCV_check(bow_clf, train_data_featurized_bow, train_labels_featurized_bow)

    # MODEL 2

    # f = open('syntax_combinations.txt', 'w')
    # f1 = open('pos_combinations.txt', 'w')
    print('Training model 2...')
    train_data_featurized = ExtractFeatures(train_data, read_from_file=True)
    print('Model 2 trained')

    # Transform labels to numeric values
    le = LabelEncoder()
    train_labels_featurized = le.fit_transform(train_labels)

    # Fit model one vs rest logistic regression
    clf = make_pipeline(DictVectorizer(), LogisticRegression())

    #########################################################################################
    # 4. TEST PREDICTIONS and ANALYSIS
    #########################################################################################

    # Fit final model on the full train data
    clf.fit(train_data_featurized, train_labels_featurized)

    print('Making predictions for the model 2...')
    # Predict on test set
    test_data, test_labels = load_data(args.test_path, verbose=False)
    test_data_featurized = ExtractFeatures(test_data)
    test_label_predicted = clf.predict(test_data_featurized)

    # Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
    test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
    print(test_label_predicted_decoded[:2])

    f = open(args.result, 'w', encoding="utf-8")
    for label in test_label_predicted_decoded:
        f.write(label + '\n')

    print('Predictions written to the file ' + args.result)

    print('CV scores for the model 2:')
    # Evaluate the model
    print(evaluateCV(clf, le, train_data_featurized, train_labels_featurized))
    evaluateCV_check(clf, train_data_featurized, train_labels_featurized)
