import itertools
import random
from collections import defaultdict
from copy import deepcopy
from typing import List

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores


def get_log_control_flow_perspective(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        for event in trace._list:
            # log_list[i].append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            log_list[i].append(event._dict.get('concept:name'))
        i += 1
    return log_list


def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        # adjust for different ngram size
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


def get_activities_to_replace(alphabet: List[str], different_activities_to_replace_count: int):
    return list(itertools.combinations(alphabet, different_activities_to_replace_count))


def get_logs_with_replaced_activities_dict(activities_to_replace_in_each_run_list, log_control_flow_perspective,
                                           different_activities_to_replace_count, activities_to_replace_with_count):
    logs_with_replaced_activities_dict = dict()
    for activities_to_replace_tuple in activities_to_replace_in_each_run_list:
        log_with_replaced_activities = deepcopy(log_control_flow_perspective)
        for trace in log_with_replaced_activities:
            i = 0
            # to replace all activites we want to replace in a trace with the same new activity we need to store the activity we choose for this trace
            activities_to_replace_with = [None] * different_activities_to_replace_count
            for activity in trace:
                if activity in activities_to_replace_tuple:
                    activity_index = activities_to_replace_tuple.index(activity)
                    if activities_to_replace_with[activity_index] is None:
                        activities_to_replace_with[activity_index] = activity + ':' + str(
                            random.randint(0, activities_to_replace_with_count - 1))
                    trace[i] = activities_to_replace_with[activity_index]
                i += 1
        logs_with_replaced_activities_dict[activities_to_replace_tuple] = log_with_replaced_activities
    return logs_with_replaced_activities_dict


def get_activity_distance_matrix_dict(activity_distance_functions, logs_with_replaced_activities_dict):
    activity_distance_matrix_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_functions:
        for key in logs_with_replaced_activities_dict:
            # Trace Distance with Levenshtein and Bose 2009
            if "Bose 2009 Substitution Scores" in activity_distance_functions:
                activity_distance_matrix = get_substitution_and_insertion_scores(
                    logs_with_replaced_activities_dict[key],
                    get_alphabet(
                        logs_with_replaced_activities_dict[
                            key]), 3)
                activity_distance_matrix_dict["Bose 2009 Substitution Scores"][key] = activity_distance_matrix
    return dict(activity_distance_matrix_dict)


def get_n_nearest_neighbors(n, replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count):
    neighbors = dict()
    for replaced_activity in replaced_activities:
        for i in range(activities_to_replace_with_count):
            replaced_activity_i = replaced_activity + ':' + str(i)
            # Collect all similarity scores for the current replaced_activity
            distances = []
            for (activity1, activity2), distance in similarity_scores_of_activities.items():
                if replaced_activity_i == activity1:
                    other_activity = activity2
                    distances.append((other_activity, distance))

            # Sort distances by the similarity score (distance)
            distances.sort(key=lambda x: x[1])

            # Get the top n nearest neighbors
            nearest_neighbors = [activity for activity, _ in distances[:n]]

            # Store the nearest neighbors in the dictionary
            neighbors[replaced_activity_i] = nearest_neighbors

    return neighbors
