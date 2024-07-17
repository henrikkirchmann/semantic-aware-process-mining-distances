import itertools
import random
from collections import defaultdict
from copy import deepcopy
from typing import List

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix


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
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


def get_activities_to_replace(alphabet: List[str], different_activities_to_replace_count: int):
    return list(itertools.combinations(alphabet, different_activities_to_replace_count))


def get_logs_with_replaced_activities_dict(activities_to_replace_in_each_run_list, log_control_flow_perspective,
                                           different_activities_to_replace_count, activities_to_replace_with_count):
    logs_with_replaced_activities_dict = dict()
    for activities_to_replace_tuple in activities_to_replace_in_each_run_list:
        #log_with_replaced_activities = deepcopy(log_control_flow_perspective)
        replacing_activities_dict = dict()
        for activity in activities_to_replace_tuple:
            replacing_activities_dict[activity] = set(range(activities_to_replace_with_count))
        log_with_replaced_activities = []
        for trace in log_control_flow_perspective:
            i = 0
            # to replace all activites we want to replace in a trace with the same new activity we need to store the activity we choose for this trace
            activities_to_replace_with = [None] * different_activities_to_replace_count
            trace_with_replaced_activities = []
            for activity in trace:
                trace_with_replaced_activities.append(activity)
                if activity in activities_to_replace_tuple:
                    activity_index = activities_to_replace_tuple.index(activity)
                    if activities_to_replace_with[activity_index] is None:
                        if len(replacing_activities_dict[activity]) == 0:
                            replacing_activities_dict[activity] = set(range(activities_to_replace_with_count))
                        activities_to_replace_with[activity_index] = activity + ':' + str(replacing_activities_dict[activity].pop())
                    trace_with_replaced_activities[i] = activities_to_replace_with[activity_index]
                i += 1
            log_with_replaced_activities.append(trace_with_replaced_activities)
        logs_with_replaced_activities_dict[activities_to_replace_tuple] = log_with_replaced_activities
    return logs_with_replaced_activities_dict


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


def get_n_nearest_neighbors(n, replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count):
    neighbors = dict()
    for replaced_activity in replaced_activities:
        for i in range(activities_to_replace_with_count):
            replaced_activity_i = replaced_activity + ':' + str(i)
            #if replaced_activity_i == "Release E:1":
            #    print("a")
            # Collect all similarity scores for the current replaced_activity
            distances = []
            for (activity1, activity2), distance in similarity_scores_of_activities.items():
                if replaced_activity_i == activity1:
                    other_activity = activity2
                    if other_activity != replaced_activity_i:
                        distances.append((other_activity, distance))

            if len(distances) > 0:
                # Sort distances by the similarity score (distance)
                distances.sort(key=lambda x: x[1], reverse=True)


                # Get the top n nearest neighbors
                nearest_neighbors = [activity for activity, _ in distances[:n]]

                # Store the nearest neighbors in the dictionary
                neighbors[replaced_activity_i] = nearest_neighbors

    return neighbors

def get_knn_dict(activity_distance_matrix_dict,
                            activities_to_replace_with_count):
    knn_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            nearest_neighbors = get_n_nearest_neighbors(1, replaced_activities,
                                                        activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count)

            knn_dict[activity_distance_function][replaced_activities] = nearest_neighbors
    return dict(knn_dict)

def get_precision_at_k(knn_dict, activity_distances):
    precision_at_k_dict = defaultdict(float)
    for activity_distance in activity_distances:
        precision_replaced_activity_at_k = 0
        for replaced_activities in knn_dict[activity_distance].keys():
            precision_at_k_sum = 0
            for replaced_activity in replaced_activities:
                for replaced_activity_with in knn_dict[activity_distance][replaced_activities].keys():
                    if replaced_activity_with[:-2] == replaced_activity:
                        precision_sum = 0
                        for activity in knn_dict[activity_distance][replaced_activities][replaced_activity_with]:
                            if activity[:-2] == replaced_activity_with[:-2]:
                                precision_sum += 1
                        a = len(knn_dict[activity_distance][replaced_activities][replaced_activity_with])
                        if a == 0:
                            print("a")
                        precision_at_k_sum += precision_sum / a
            precision_replaced_activity_at_k += precision_at_k_sum / len(knn_dict[activity_distance][replaced_activities].keys())
        precision_at_k_dict[activity_distance] = precision_replaced_activity_at_k / len(knn_dict[activity_distance].keys())
    return dict(precision_at_k_dict)