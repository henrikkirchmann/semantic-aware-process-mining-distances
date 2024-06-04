from collections import defaultdict
from typing import List, Tuple, Dict
from math import comb, log2
from data_util import algorithm
import pm4py
import math
from pm4py.objects.log.importer.xes import importer as xes_importer


def transformLogToTraceStringList(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        for event in trace._list:
            log_list[i].append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            #log_list[i].append(event._dict.get('concept:name'))

        i += 1
    return log_list

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


def get_common_context_dict(context_dict: Dict[str, Dict[Tuple[str, ...], int]]) -> Dict[
    Tuple[str, str], List[Tuple[str, ...]]]:
    common_context_dict = defaultdict(list)

    grams = list(context_dict.keys())
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            gram1 = grams[i]
            gram2 = grams[j]

            contexts1 = set(context_dict[gram1].keys())
            contexts2 = set(context_dict[gram2].keys())

            common_contexts = contexts1.intersection(contexts2)
            if common_contexts:
                common_context_dict[(gram1, gram2)] = list(common_contexts)

    return dict(common_context_dict)

def get_cooccurrence_counts(context_dict: Dict[Tuple[str, ...], int], common_context_dict: Dict[Tuple[str, str], List[Tuple[str, ...]]], unique_activities) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]:
    cooccurrence_counts_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for co-occurance frequencies
    #co-occurrence combinations is for ‘like’ acitivities
    for activity in context_dict:
        for entry in context_dict[activity]:
            cooccurrence_counts_dict[entry][(activity, activity)] = comb(context_dict[activity].get(entry, 0), 2) # n over 2
    #co-occurrence combinations is for ‘un-like’ acitivities
    for activity_pair in common_context_dict.keys():
        for cooccurrence in common_context_dict[activity_pair]:
            cooccurrence_counts_dict[cooccurrence][activity_pair] = context_dict[activity_pair[0]][cooccurrence] * context_dict[activity_pair[1]][cooccurrence]
    return {k: dict(v) for k, v in cooccurrence_counts_dict.items()}  # Convert inner defaultdicts to regular dicts

def get_normalized_co_occurrence_counts(cooccurrence_counts:Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]) -> Dict[Tuple[str, str], float]:
    normalized_co_occurrence_counts_dict = defaultdict(float)
    sum_of_cooccurrence_combinations_of_allcontext_dict = defaultdict(float)
    all_cooccurrence_set = set()
    for activity_pair in cooccurrence_counts.keys():
        for cooccurrence in list(cooccurrence_counts[activity_pair].keys()):
            all_cooccurrence_set.add(cooccurrence)
    for activity_pair in all_cooccurrence_set:
        sum = 0
        for cooccurrence in cooccurrence_counts.keys():
            sum += cooccurrence_counts[cooccurrence].get(activity_pair, 0)
        sum_of_cooccurrence_combinations_of_allcontext_dict[activity_pair] = sum
    sum_of_all = 0
    for value in sum_of_cooccurrence_combinations_of_allcontext_dict.values():
        sum_of_all += value
    for activity_pair in sum_of_cooccurrence_combinations_of_allcontext_dict.keys():
        normalized_co_occurrence_counts_dict[activity_pair] = sum_of_cooccurrence_combinations_of_allcontext_dict[activity_pair]/sum_of_all
    return dict(normalized_co_occurrence_counts_dict)

def get_probabilities_of_symbol_occurrence(sum_of_cooccurrence_combinations_of_allcontext_dict: Dict[Tuple[str, str], float], unique_activities: List[str]) -> Dict[str, float]:
    probabilities_of_symbol_occurrence = defaultdict(float)
    for activity in unique_activities:
        probability = sum_of_cooccurrence_combinations_of_allcontext_dict.get((activity,activity), 0.0)
        for activity_different in unique_activities:
            if activity != activity_different:
                probability += sum_of_cooccurrence_combinations_of_allcontext_dict.get((activity, activity_different), 0.0)
        probabilities_of_symbol_occurrence[activity] = probability
    return dict(probabilities_of_symbol_occurrence)

def get_substitution_scores(probabilities_of_symbol_occurrence, unique_activities: List[str], sum_of_cooccurrence_combinations_of_allcontext_dict) -> Dict[Tuple[str, str], float]:
    substitution_scores = defaultdict(float)
    for activity_1 in unique_activities:
        for activity_2 in unique_activities:
            if activity_1 == activity_2:
                expected_value = pow(probabilities_of_symbol_occurrence.get(activity_1, 0.0), 2)
            else:
                expected_value = 2 * probabilities_of_symbol_occurrence[activity_1] * probabilities_of_symbol_occurrence[activity_2]
            sum_of_cooccurrence = sum_of_cooccurrence_combinations_of_allcontext_dict.get((activity_1, activity_2), 0.0)
            if expected_value == 0.0 or sum_of_cooccurrence == 0.0:
                substitution_scores[(activity_1, activity_2)] = 0.0
            else:
                substitution_scores[(activity_1, activity_2)] = log2(sum_of_cooccurrence/expected_value)
    return dict(substitution_scores)



# Example usage
log = xes_importer.apply('repairExample.xes')
log = transformLogToTraceStringList(log)
#log = [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c'], ['a', 'd', 'c'],  ['a', 'b', 'c'], ['a', 'd', 'c']]
#log = [['a', 'a', 'b', 'c', 'd', 'b', 'b', 'c', 'd', 'a'], ['d', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'b', 'b'], ['b', 'b', 'b', 'c', 'd', 'b', 'b', 'b', 'c', 'c', 'a', 'a'], ['a', 'a', 'a', 'd', 'a', 'b', 'b', 'c', 'c', 'c'], ['a', 'a', 'a', 'c', 'd', 'c', 'd', 'c', 'b', 'e', 'd', 'b', 'c', 'c', 'b', 'a', 'd', 'b', 'd', 'e', 'b', 'd', 'c']]

#Algorithm 1: Algorithm to derive substitution scores
#Line 1
ngram_size = 3
unique_activity_set = algorithm.get_all_activities_from_list_of_traces(log)
#Line 2
ngrams_dict = get_ngrams_dict(log, ngram_size)
print(ngrams_dict)
#Line 3
context_dict = get_context_dict(ngrams_dict)
print(context_dict)
#Line 4
common_context_dict = get_common_context_dict(context_dict)
print(common_context_dict)
#Line 5
cooccurrence_counts = get_cooccurrence_counts(context_dict, common_context_dict, unique_activity_set)
print(cooccurrence_counts)
#Line 6-8
sum_of_cooccurrence_combinations_of_allcontext_dict = get_normalized_co_occurrence_counts(cooccurrence_counts)
print(sum_of_cooccurrence_combinations_of_allcontext_dict)
#Line 9
probabilities_of_symbol_occurrence = get_probabilities_of_symbol_occurrence(sum_of_cooccurrence_combinations_of_allcontext_dict, unique_activity_set)
print(probabilities_of_symbol_occurrence)
#Line 10-11
substitution_scores = get_substitution_scores(probabilities_of_symbol_occurrence, unique_activity_set, sum_of_cooccurrence_combinations_of_allcontext_dict)
print(substitution_scores)
print(substitution_scores.get(('Repair (Simple)-start', 'Repair (Complex)-start')))
print(substitution_scores.get(('Repair (Simple)-start', 'Repair (Simple)-start')))
print(substitution_scores.get(('Test Repair-complete', 'Archive Repair-complete')))
print(substitution_scores.get(('Inform User-complete', 'Archive Repair-complete')))

#print(substitution_scores.get(('Repair (Simple)', 'Repair (Complex)')))
#print(substitution_scores.get(('Repair (Simple)', 'Repair (Simple)')))
#print(substitution_scores.get(('Test Repair', 'Archive Repair')))
#print(substitution_scores.get(('Inform User', 'Archive Repair')))



"""
Expected output:
{
    ('b', 'c'): [('a', 'c')],
    ('c', 'd'): [('b', 'c')]
}

Explanation:
- The input list of lists is [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c']].
- For ngram size of 3:
  - In the first sublist ['a', 'b', 'c', 'd']:
    - The n-grams are ('a', 'b', 'c') and ('b', 'c', 'd').
  - In the second sublist ['b', 'c', 'd', 'e']:
    - The n-grams are ('b', 'c', 'd') and ('c', 'd', 'e').
  - In the third sublist ['a', 'b', 'c']:
    - The n-gram is ('a', 'b', 'c').
- The n-gram counts are:
  {('a', 'b', 'c'): 2, ('b', 'c', 'd'): 2, ('c', 'd', 'e'): 1}.
- The context dictionary is:
  - 'b' is the middle gram in ('a', 'b', 'c') and occurs twice with context ('a', 'c').
  - 'c' is the middle gram in ('b', 'c', 'd') and occurs twice with context ('b', 'd').
  - 'd' is the middle gram in ('c', 'd', 'e') with context ('c', 'e') once and in ('b', 'c') with context ('b', 'c') once.
- Common contexts are:
  - 'b' and 'c' share context ('a', 'c').
  - 'c' and 'd' share context ('b', 'c').
"""
"""
log = [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e'], ['a', 'b', 'c'], ['a', 'd', 'c']]
ngram_size = 3
ngrams_dict = get_ngrams_dict(log, ngram_size)
ngrams_dict: 
{
    ('a', 'b', 'c'): 2,
    ('b', 'c', 'd'): 2,
    ('c', 'd', 'e'): 1,
    ('a', 'd', 'c'): 1
}
context_dict = get_context_dict(ngrams_dict)
context_dict: 
{
    'b': {('a', 'c'): 2},
    'c': {('b', 'd'): 2},
    'd': {('c', 'e'): 1, ('a', 'c'): 1}
}
common_context_dict = get_common_context_dict(context_dict)
{
    ('b', 'd'): [('a', 'c')] # activity b and d both happend as "abc" and "adc"
}
"""