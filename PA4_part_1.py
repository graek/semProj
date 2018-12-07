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
from sklearn.preprocessing import FunctionTransformer,LabelEncoder
import numpy as np

############################################################################################
# 1. LOAD DATA
############################################################################################

PairExample = namedtuple('PairExample',
    'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
    'left, mention_1, middle, mention_2, right, direction')
def load_data(file, verbose=True):
    f = open(file,'r', encoding='utf-8')
    data = []
    labels = []
    for i,line in enumerate(f):
        instance = json.loads(line)
        if i==0:
            if verbose:
                print('json example:')
                print(instance)
        #'relation, entity_1, entity_2, snippet' fileds for each example
        #'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'],instance['entity_2'],[])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'],snippet['mention_1'],
                                        snippet['middle'], 
                                        snippet['mention_2'],snippet['right'],
                                        snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i==0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])
    return data,labels
    
train_data, train_labels = load_data('train.json.txt')

###########################################################################################
# 2. EXTRACT FEATURES and BUILD CLASSIFIER
###########################################################################################

# Extract two simple features
def ExractSimpleFeatures(data, verbose=True):
    featurized_data = []
    for instance in data:
        bow = set()
        for s in instance.snippet:
            bow.update(s.left.split(), s.middle.split(), s.right.split())
        featurized_data.append(' '.join(bow))
    return featurized_data

# Transform dataset to features
train_data_featurized = ExractSimpleFeatures(train_data)

# Transform labels to numeric values
le = LabelEncoder()
train_labels_featurized = le.fit_transform(train_labels)

# Fit model one vs rest logistic regression    
bow_clf = make_pipeline(CountVectorizer(), LogisticRegression())

#########################################################################################
# 4. TEST PREDICTIONS and ANALYSIS
#########################################################################################

# Fit final model on the full train data
bow_clf.fit(train_data_featurized, train_labels_featurized)

# Predict on test set
test_data, test_labels = load_data('test.json.txt', verbose=False)
test_data_featurized = ExractSimpleFeatures(test_data, verbose=False)
test_label_predicted = bow_clf.predict(test_data_featurized)

# Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
print(test_label_predicted_decoded[:2])

f = open("test_labels.txt", 'w', encoding="utf-8")
for label in test_label_predicted_decoded:
    f.write(label+'\n')