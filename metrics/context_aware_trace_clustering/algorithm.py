from collections import defaultdict
from typing import List, Tuple, Dict
import os

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from data_util.algorithm import transform_log_to_trace_string_list_with_padding, get_all_activities_from_list_of_traces
from metrics.context_aware_trace_clustering.substitution_scores import get_substitution_scores
from metrics.context_aware_trace_clustering.insertion_scores import get_insertion_scores

def get_ngrams_dict(log: List[List[str]], ngram_size: int) -> Dict[Tuple[str, ...], int]:
    ngrams_dict = defaultdict(int)  # Using defaultdict to handle counting

    for sublist in log:
        for i in range(len(sublist) - ngram_size + 1):
            ngram = tuple(sublist[i:i + ngram_size])  # Convert the n-gram to a tuple to use as a dictionary key
            ngrams_dict[ngram] += 1  # Increment the count for this n-gram

    return dict(ngrams_dict)  # Convert back to a regular dictionary if desired

def get_context_dict(ngrams_dict: Dict[Tuple[str, ...], int]) -> Dict[str, Dict[Tuple[str, ...], int]]:
    context_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for context frequencies

    for ngram, count in ngrams_dict.items():
        middle_index = len(ngram) // 2
        middle_gram = ngram[middle_index]
        # Create context by removing the middle element
        context_before = ngram[:middle_index]
        context_after = ngram[middle_index + 1:]
        surrounding_grams = context_before + context_after

        context_dict[middle_gram][surrounding_grams] += count

    return {k: dict(v) for k, v in context_dict.items()}  # Convert inner defaultdicts to regular dicts

log = xes_importer.apply('/Users/henrikkirchmann/Documents/I2NLP/semantic-aware-metric/repairExample.xes')
#pt = pm4py.discover_process_tree_inductive(log)
#pm4py.view_process_tree(pt)

log = transform_log_to_trace_string_list_with_padding(log)
#log = [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c'], ['a', 'd', 'c'],  ['a', 'b', 'c'], ['a', 'd', 'c']]
#log = [['a', 'a', 'b', 'c', 'd', 'b', 'b', 'c', 'd', 'a'], ['d', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'b', 'b'], ['b', 'b', 'b', 'c', 'd', 'b', 'b', 'b', 'c', 'c', 'a', 'a'], ['a', 'a', 'a', 'd', 'a', 'b', 'b', 'c', 'c', 'c'], ['a', 'a', 'a', 'c', 'd', 'c', 'd', 'c', 'b', 'e', 'd', 'b', 'c', 'c', 'b', 'a', 'd', 'b', 'd', 'e', 'b', 'd', 'c']]

ngram_size = 3
#Line 1 in Algorithm 1 & 2
alphabet = get_all_activities_from_list_of_traces(log)
#Line 2 in Algoirthm 1 & 2
ngrams_dict = get_ngrams_dict(log, ngram_size)
#Line 3 in Algoirthm 1 & 2
context_dict = get_context_dict(ngrams_dict)

#Algorithm 1: Algorithm to derive substitution scores
substitution_scores = get_substitution_scores(alphabet, context_dict)
#Algorithm 2: Algorithm to derive insertion scores
#insertion_scores = get_insertion_scores(alphabet, context_dict, ngram_size)

print(substitution_scores)
if 1:
    print(str(substitution_scores.get(('Repair (Simple)-start', 'Repair (Complex)-start'))) + ' vs 5')
    print(str(substitution_scores.get(('Repair (Simple)-start', 'Repair (Simple)-start'))) + ' vs 9')
    print(str(substitution_scores.get(('Test Repair-complete', 'Archive Repair-complete'))) + ' vs -11')
    print(str(substitution_scores.get(('Inform User-complete', 'Archive Repair-complete'))) + ' vs 0')
if 0:
    print(str(substitution_scores.get(('Repair (Simple)', 'Repair (Complex)'))) + ' vs 5')
    print(str(substitution_scores.get(('Repair (Simple)', 'Repair (Simple)'))) + ' vs 9')
    print(str(substitution_scores.get(('Test Repair', 'Archive Repair'))) + ' vs -11')
    print(str(substitution_scores.get(('Inform User', 'Archive Repair'))) + ' vs 0')
