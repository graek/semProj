import json
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.metrics import confusion_matrix

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

def get_prediction(f):
    y_prediction = []
    for line in f:
        y_prediction.append(line.strip('\n'))
    return y_prediction


PairExample = namedtuple('PairExample',
                             'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
                         'left, mention_1, middle, mention_2, right, direction')

gold_data, gold_labels = load_data('gold.json.txt', verbose=False)

f = open('test_labels.txt', 'r')

y_true = gold_labels
y_prediction = get_prediction(f)

cnf_matrix = confusion_matrix(y_true, y_prediction)

def process_cm(confusion_mat, i=0, to_print=True):
    # i means which class to choose to do one-vs-the-rest calculation
    # rows are actual obs whereas columns are predictions
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    if to_print:
        print('True Positive: {}'.format(TP))
        print('False Positive: {}'.format(FP))
        print('False Negative: {}'.format(FN))
        print('True Negative: {}'.format(TN))
    return TP, FP, FN, TN


pd.set_option('display.max_columns', 7)
unique_label = np.unique(y_true)
print(pd.DataFrame(confusion_matrix(y_true, y_prediction, labels=unique_label),
                   index=['true:{:}'.format(x) for x in unique_label],
                   columns=['pred:{:}'.format(x) for x in unique_label]))

counter = 1
print('\n')
for i in ['author', 'capital', 'has_spouse', 'worked_at']:
    print('Contigency table for label %s' % i)
    process_cm(cnf_matrix, counter, to_print=True)
    print('\n')
    counter += 1