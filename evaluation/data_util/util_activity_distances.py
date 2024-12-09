from collections import defaultdict
from typing import List
import sys
import psutil
import math
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
import gc

def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


def get_activity_distance_matrix_dict_list(args):
    log_control_flow_perspective, activity_distance_function, alphabet = args

    n_gram_size_bose_2009 = 3

    get_distance_matrix = get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function,
                                                       alphabet, n_gram_size_bose_2009)

    # Step 1: Find the minimum and maximum values
    min_value = min(get_distance_matrix[activity_distance_function].values())
    max_value = max(get_distance_matrix[activity_distance_function].values())

    # Step 2: Normalize the values
    normalized_data = {key: (value - min_value) / (max_value - min_value) for key, value in
                       get_distance_matrix[activity_distance_function].items()}

    distance_dict = dict()
    distance_dict[activity_distance_function] = normalized_data

    return distance_dict


def get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function, alphabet,
                                 n_gram_size_bose_2009=3):
    activity_distance_matrix_dict = defaultdict()
    if "Bose 2009 Substitution Scores" == activity_distance_function:
        activity_distance_matrix = get_substitution_and_insertion_scores(
            log_control_flow_perspective,
            alphabet, n_gram_size_bose_2009)
        activity_distance_matrix_dict[activity_distance_function] = activity_distance_matrix
    elif "De Koninck 2018 act2vec" == activity_distance_function[:23]:
        if activity_distance_function[24:] == "CBOW":
            sg = 0
        else:
            sg = 1
        act2vec_distance_matrix = get_act2vec_distance_matrix(log_control_flow_perspective,
                                                              alphabet, sg)
        activity_distance_matrix_dict[activity_distance_function] = act2vec_distance_matrix
    return dict(activity_distance_matrix_dict)


def get_activity_distance_matrix_dict(activity_distance_functions, logs_with_replaced_activities_dict,
                                      n_gram_size_bose_2009=3):
    activity_distance_matrix_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_functions:
        if "Bose 2009 Substitution Scores" == activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                activity_distance_matrix = get_substitution_and_insertion_scores(
                    logs_with_replaced_activities_dict[key],
                    get_alphabet(
                        logs_with_replaced_activities_dict[
                            key]), n_gram_size_bose_2009)
                activity_distance_matrix_dict[activity_distance_function][key] = activity_distance_matrix
        elif "De Koninck 2018 act2vec" == activity_distance_function[:23]:
            if activity_distance_function[24:] == "CBOW":
                sg = 0
            else:
                sg = 1
            for key in logs_with_replaced_activities_dict:
                act2vec_distance_matrix = get_act2vec_distance_matrix(logs_with_replaced_activities_dict[key],
                                                                      get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]), sg)
                activity_distance_matrix_dict[activity_distance_function][key] = act2vec_distance_matrix
        elif "Unit Distance" == activity_distance_function:
            for key in logs_with_replaced_activities_dict:
                unit_distance_matrix = get_unit_cost_activity_distance_matrix(logs_with_replaced_activities_dict[key], get_alphabet(
                                                                          logs_with_replaced_activities_dict[key]))
                activity_distance_matrix_dict[activity_distance_function][key] = unit_distance_matrix




    return dict(activity_distance_matrix_dict)


def get_log_control_flow_perspective_with_short_activity_names(log_control_flow_perspective, alphabet):
    activity_names_dict = dict()
    i = 0
    for activity in alphabet:
        activity_names_dict[activity] = str(i)
        i += 1
    for trace in log_control_flow_perspective:
        i = 0
        for activity in trace:
            trace[i] = activity_names_dict[activity]
            i += 1
    return log_control_flow_perspective


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def unresponsiveness_prediction(log_size, alphabet_size, r, w, sampling_size=None):
    memory_info = psutil.virtual_memory()
    # Get the amount of free memory
    free_memory = memory_info.free

    # Initialize the sum
    total_sum = 0
    for r_val in range(1, r + 1):
        inner_sum = 0
        for w_val in range(2, w + 1):
            # Calculate (alphabet_size choose r_val) * log_size
            combination = math.comb(alphabet_size, r_val)
            if sampling_size is not None:
                combination = sampling_size
            inner_sum += combination * log_size
        total_sum += inner_sum

    if total_sum > free_memory/2:
        #print("System might run out of memory.")
        return True
    else:
        return False


def print_log_stats(log, alphabet):
    print("Alphabet Size: " + str(len(alphabet)))
    print("Number of Traces: " + str(len(log)))

    # Convert each list to a tuple and then use a set to find unique tuples
    trace_variants = set(tuple(trace) for trace in log)

    # Get the number of unique lists
    print("Number of Trace Variants: " + str(len(trace_variants)))

    # Get the lengths of all lists
    lengths = [len(trace) for trace in log]

    # Calculate min, max, and average length
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)

    print("Trace Length Min: " + str(min_length))
    print("Trace Length Avg: " + str(avg_length))
    print("Trace Length Max: " + str(max_length))


def get_unit_cost_activity_distance_matrix(log, alphabet):
    distances = {}
    for activity1 in alphabet:
        for activity2 in alphabet:
            distances[(activity1, activity2)] = 1
    return distances




