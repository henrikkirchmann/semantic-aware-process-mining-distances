from typing import List
from collections import defaultdict
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix

def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)

def get_activity_distance_matrix_dict_list(args):
    log_control_flow_perspective, activity_distance_function, alphabet = args

    n_gram_size_bose_2009 = 3

    get_distance_matrix = get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function, alphabet, n_gram_size_bose_2009)

    # Step 1: Find the minimum and maximum values
    min_value = min(get_distance_matrix[activity_distance_function].values())
    max_value = max(get_distance_matrix[activity_distance_function].values())

    # Step 2: Normalize the values
    normalized_data = {key: (value - min_value) / (max_value - min_value) for key, value in get_distance_matrix[activity_distance_function].items()}

    distance_dict = dict()
    distance_dict[activity_distance_function] = normalized_data

    return distance_dict

def get_activity_distance_matrix(log_control_flow_perspective, activity_distance_function, alphabet, n_gram_size_bose_2009 = 3):
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

def get_activity_distance_matrix_dict(activity_distance_functions, logs_with_replaced_activities_dict, n_gram_size_bose_2009 = 3):
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
                        get_alphabet(logs_with_replaced_activities_dict[key]), sg)
                    activity_distance_matrix_dict[activity_distance_function][key] = act2vec_distance_matrix
    return dict(activity_distance_matrix_dict)




