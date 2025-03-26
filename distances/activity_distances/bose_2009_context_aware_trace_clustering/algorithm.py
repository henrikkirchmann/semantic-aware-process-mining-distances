from collections import defaultdict
from typing import List, Tuple, Dict

from distances.activity_distances.data_util.algorithm import give_log_padding, get_ngrams_dict, get_context_dict
from distances.activity_distances.bose_2009_context_aware_trace_clustering.substitution_scores import get_substitution_scores
import numpy as np

def get_substitution_and_insertion_scores(log, alphabet, ngram_size):
    log = give_log_padding(log, ngram_size)

    #Line 2 in Algoirthm 1 & 2
    ngrams_dict = get_ngrams_dict(log, ngram_size)
    #Line 3 in Algoirthm 1 & 2
    context_dict = get_context_dict(ngrams_dict)

    #Algorithm 1: Algorithm to derive substitution scores
    substitution_scores, probabilities_of_symbol_occurrence  = get_substitution_scores(alphabet, context_dict)
    #Algorithm 2: Algorithm to derive insertion scores
    #insertion_scores = get_insertion_scores(alphabet, context_dict, ngram_size, probabilities_of_symbol_occurrence)
    # 1. Extract unique row labels (and assume the same set applies for columns)
    keys = sorted({key[0] for key in substitution_scores.keys()})
    # Optionally, verify that the same set exists in the second element of each key tuple:
    assert keys == sorted({key[1] for key in substitution_scores.keys()}), "Matrix is not square."

    # 2. Build the embedding dictionary where each row maps to its corresponding numpy array.
    embedding_dict = {
        row: np.array([substitution_scores[(row, col)] for col in keys])
        for row in keys
    }
    return substitution_scores, embedding_dict #, insertion_scores

#print(substitution_scores)
#if 1:
#    print(str(substitution_scores.get(('Repair (Simple)-start', 'Repair (Complex)-start'))) + ' vs 5')
#    print(str(substitution_scores.get(('Repair (Simple)-start', 'Repair (Simple)-start'))) + ' vs 9')
#    print(str(substitution_scores.get(('Test Repair-complete', 'Archive Repair-complete'))) + ' vs -11')
#    print(str(substitution_scores.get(('Inform User-complete', 'Archive Repair-complete'))) + ' vs 0')
#if 0:
#    print(str(substitution_scores.get(('Repair (Simple)', 'Repair (Complex)'))) + ' vs 5')
#    print(str(substitution_scores.get(('Repair (Simple)', 'Repair (Simple)'))) + ' vs 9')
#    print(str(substitution_scores.get(('Test Repair', 'Archive Repair'))) + ' vs -11')
#    print(str(substitution_scores.get(('Inform User', 'Archive Repair'))) + ' vs 0')
