############################################################################################
# GROUP MEMBERS:
# Lukas Grässle
# Olga Sozinova
# Xiao'ao Song
# Yue Ding
############################################################################################

import gzip
import numpy as np
import random
import os
import json

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

PairExample = namedtuple('PairExample',
                         'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
                     'left, mention_1, middle, mention_2, right, direction')


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


train_data, train_labels = load_data('train.json.txt')


###########################################################################################
# 2. EXTRACT FEATURES and BUILD CLASSIFIER
###########################################################################################

# Extract features
def ExtractFeatures(data, read_from_file=True, verbose=True):
    counter = 1
    featurized_data = []
    nlp = spacy.load('en')

    if read_from_file:
        f1 = open('pos_combinations.txt', 'r')
        for line in f1:
            featurized_data.append(line)
    else:
        for instance in data:
            #bow = set()
            current_syntax = []
            current_pos = []
            for s in instance.snippet:
                #bow.update(s.left.split(), s.middle.split(), s.right.split())
                doc = nlp(s.middle)
                syntax_combination = syntax_features(doc)
                current_syntax.append(syntax_combination)

                pos_combination = pos_tags(doc)
                current_pos.append(pos_combination)

            #featurized_data.append(' '.join(bow))
            featurized_data.append(' '.join(current_pos))
            #f.write(' '.join(current_syntax) + '\n')
            #f1.write(' '.join(current_pos) + '\n')
            #print(counter, ' '.join(current_pos))
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
def delete_punct(doc):
    output_line = []
    for token in doc:
        if not token.is_punct:
            output_line.append(token)
    return ' '.join(output_line)

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

#f = open('syntax_combinations.txt', 'w')
#f1 = open('pos_combinations.txt', 'w')

# Transform dataset to features
train_data_featurized = ExtractFeatures(train_data, read_from_file=True)

# Transform labels to numeric values
le = LabelEncoder()
train_labels_featurized = le.fit_transform(train_labels)

# Fit model one vs rest logistic regression
clf = make_pipeline(CountVectorizer(), LogisticRegression())

#########################################################################################
# 4. TEST PREDICTIONS and ANALYSIS
#########################################################################################

# Fit final model on the full train data
clf.fit(train_data_featurized, train_labels_featurized)

# Predict on test set
test_data, test_labels = load_data('test.json.txt', verbose=False)
test_data_featurized = ExtractFeatures(test_data, read_from_file=False, verbose=False)
test_label_predicted = clf.predict(test_data_featurized)

# Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
print(test_label_predicted_decoded[:2])

# Evaluate the model
print(evaluateCV(clf, le, test_data_featurized, test_label_predicted))
evaluateCV_check(clf, test_data_featurized, test_label_predicted)

f = open("test_labels.txt", 'w', encoding="utf-8")
for label in test_label_predicted_decoded:
    f.write(label + '\n')
