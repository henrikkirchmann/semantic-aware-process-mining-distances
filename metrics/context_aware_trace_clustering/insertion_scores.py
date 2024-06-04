from typing import List, Tuple, Dict
from collections import defaultdict
from math import comb

def get_same_symbols_cooccurrence_counts(context_dict: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], int]]:
    cooccurrence_counts_dict = defaultdict(lambda: defaultdict(int))  # Nested defaultdict for co-occurance frequencies
    #co-occurrence combinations is for ‘like’ acitivities
    for activity in context_dict:
        for entry in context_dict[activity]:
            cooccurrence_counts_dict[entry][(activity, activity)] = comb(context_dict[activity].get(entry, 0), 2) # n over 2
    return {k: dict(v) for k, v in cooccurrence_counts_dict.items()}  # Convert inner defaultdicts to regular dicts

def get_right_given_left_count(alphabet, context_dict, same_symbols_cooccurrence_counts_dict, ngram_size):
    right_given_left_count = defaultdict(lambda: defaultdict(int))
    middle_index = ngram_size // 2
    for activity_1 in alphabet:
        for activity_2 in alphabet:
            for activity_2_with_y in context_dict[activity_1].keys():
                if activity_2_with_y not in context_dict[activity_2].keys():
                    y = activity_2_with_y[middle_index:]


            right_given_left_count[activity_1][activity_2] += 1


def get_insertion_scores(alphabet, context_dict, ngram_size):
    #Line 4
    same_symbols_cooccurrence_counts_dict = get_same_symbols_cooccurrence_counts(context_dict)
    #Line 5
    right_given_left_count_dict = get_right_given_left_count(alphabet, context_dict, same_symbols_cooccurrence_counts_dict, ngram_size)
    return