import itertools
from collections import defaultdict
from pathlib import Path
from typing import List
import gc
import copy
import random
import math

from networkx.algorithms.approximation import diameter

from evaluation.data_util.util_activity_distances import get_obj_size
import pandas as pd
from definitions import ROOT_DIR
import sys
from itertools import combinations


def get_log_control_flow_perspective(log):
    log_list = list()
    for trace in log:
        trace_list = list()
        for event in trace._list:
            #trace_list.append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            trace_list.append(event._dict.get('concept:name'))
        for _ in range(1, 1+1):
            log_list.append(copy.copy(trace_list))
    return log_list

def reservoir_sampling(iterator, sample_size):
    """Perform reservoir sampling from the given iterator."""
    reservoir = []
    for index, item in enumerate(iterator):
        if len(reservoir) < sample_size:
            reservoir.append(item)
        else:
            # Randomly replace elements in the reservoir with decreasing probability
            replace_index = random.randint(0, index)
            if replace_index < sample_size:
                reservoir[replace_index] = item
    return reservoir


def get_activities_to_replace(alphabet: List[str], different_activities_to_replace_count: int, sample_size: int):
    alphabet_len = len(alphabet)
    alphabet_len_minus_one = alphabet_len -1
    sampled_combinations = set()
    comb_number = math.comb(alphabet_len, different_activities_to_replace_count)
    while len(sampled_combinations) < sample_size and len(sampled_combinations) < comb_number:
        activity_index_set = set()
        while len(activity_index_set) < different_activities_to_replace_count:
            activity_index_set.add(random.randint(0, alphabet_len_minus_one))
        activity_index_list = list(activity_index_set)
        activity_index_list.sort()
        acitvity_list = list()
        for index in activity_index_list:
            acitvity_list.append(alphabet[index])
        sampled_combinations.add(tuple(acitvity_list))
    return sampled_combinations


def get_logs_with_replaced_activities_dict(activities_to_replace_in_each_run_list, log_control_flow_perspective,
                                           different_activities_to_replace_count, activities_to_replace_with_count):
    logs_with_replaced_activities_dict = dict()
    for activities_to_replace_tuple in activities_to_replace_in_each_run_list:
        # log_with_replaced_activities = deepcopy(log_control_flow_perspective)
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
                if activity in replacing_activities_dict:
                    activity_index = activities_to_replace_tuple.index(activity)
                    if activities_to_replace_with[activity_index] is None:
                        if len(replacing_activities_dict[activity]) == 0:
                            replacing_activities_dict[activity] = set(range(activities_to_replace_with_count))
                        activities_to_replace_with[activity_index] = activity + ':' + str(
                            replacing_activities_dict[activity].pop())
                    trace_with_replaced_activities[i] = activities_to_replace_with[activity_index]
                i += 1
            log_with_replaced_activities.append(trace_with_replaced_activities)
        logs_with_replaced_activities_dict[activities_to_replace_tuple] = log_with_replaced_activities
    return logs_with_replaced_activities_dict


def get_n_nearest_neighbors(n, replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse):
    neighbors = dict()
    similarity_scores_of_activities = get_nested_dict(similarity_scores_of_activities)
    for replaced_activity in replaced_activities:
        for i in range(activities_to_replace_with_count):
            replaced_activity_i = replaced_activity + ':' + str(i)
            # Collect all similarity scores for the current replaced_activity
            distances = []

            for activity in similarity_scores_of_activities[replaced_activity_i]:
                if activity != replaced_activity_i:
                    distances.append((activity, similarity_scores_of_activities[replaced_activity_i][activity]))

            if len(distances) > 0:
                # Sort distances by the similarity score (distance)
                distances.sort(key=lambda x: x[1], reverse=reverse)

                # Get the top n nearest neighbors
                nearest_neighbors = [activity for activity, _ in distances[:n]]

                # Store the nearest neighbors in the dictionary
                neighbors[replaced_activity_i] = nearest_neighbors

    return neighbors


def get_knn_dict(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, knn_count):
    knn_dict = defaultdict(lambda: defaultdict())
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            nearest_neighbors = get_n_nearest_neighbors(knn_count, replaced_activities,
                                                        activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count,
                                                        reverse)

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
                    replaced_activity_with_split = replaced_activity_with.split(':')
                    if not replaced_activity_with_split[1].isdigit():
                        sys.exit("Naming Error")
                    if replaced_activity_with_split[0] == replaced_activity:
                        precision_sum = 0
                        for activity in knn_dict[activity_distance][replaced_activities][replaced_activity_with]:
                            activity_split = activity.split(':')
                            if activity_split[0] == replaced_activity_with_split[0]:
                                precision_sum += 1
                        a = len(knn_dict[activity_distance][replaced_activities][replaced_activity_with])
                        precision_at_k_sum += precision_sum / a
            precision_replaced_activity_at_k += precision_at_k_sum / len(
                knn_dict[activity_distance][replaced_activities].keys())
        precision_at_k_dict[activity_distance] = precision_replaced_activity_at_k / len(
            knn_dict[activity_distance].keys())
    return dict(precision_at_k_dict)


def save_intrinsic_results(activity_distance_functions, results, log_name, r, w, sampling_size):
    activity_distance_function_index = 0
    for activity_distance_function in activity_distance_functions:
        results_per_activity_distance_function = list()
        for result in results:
            results_per_activity_distance_function.append(result[activity_distance_function_index])

        # Create DataFrame from results
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/" + log_name).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results_per_activity_distance_function, columns=['r', 'w', 'diameter', 'precision@w-1', 'precision@1', 'triplet'])

        # Generate the file name incorporating all function arguments if no output_file is provided
        csv = f"{log_name}_distfunc_{activity_distance_function}_r{r}_w{w}_samplesize_{sampling_size}.csv"

        # Ensure that the file name is valid and does not contain invalid characters (especially for file systems)
        csv = csv.replace("/", "_").replace("\\", "_")

        # Save the DataFrame to a CSV file
        df.to_csv(ROOT_DIR + "/results/activity_distances/intrinsic/" + log_name + "/"+ csv, index=False)

        # Print average results
        print_average_results(df, activity_distance_function)

        activity_distance_function_index += 1


       # visualization_intrinsic_evaluation(results_per_activity_distance_function, activity_distance_function,
       #                                    log_name, r, w, sampling_size, output_file)
       # load_visualization_intrinsic_evaluation(results_per_activity_distance_function, activity_distance_function,
       #                                         log_name, r, w, sampling_size, output_file)
       # activity_distance_function_index = activity_distance_function_index + 1

def print_average_results(df, activity_distance_function):

    #diameter
    result = df.pivot(index='w', columns='r', values='diameter')
    average_value = result.values.mean()
    print("The average diameter is: " + str(average_value) + " " + activity_distance_function)


    #precision@w-1
    result = df.pivot(index='w', columns='r', values='precision@w-1')
    average_value = result.values.mean()
    print("The average precision@w-1 is: " + str(average_value) + " " + activity_distance_function)

    #precision@1
    result = df.pivot(index='w', columns='r', values='precision@1')
    average_value = result.values.mean()
    print("The average Nearest Neighbor is: " + str(average_value) + " " + activity_distance_function)

    #precision@1
    result = df.pivot(index='w', columns='r', values='triplet')
    average_value = result.values.mean()
    print("The average triplet value is: " + str(average_value) + " " + activity_distance_function)


#'''
def get_triplet(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, alphabet):
    triplet_list = list()
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            avg_triplet_value = get_avg_triplet_value(replaced_activities, activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count, reverse, alphabet)
            triplet_list.append(avg_triplet_value)

    return sum(triplet_list) / len(triplet_list)

def get_avg_triplet_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse, alphabet):
    #high bose similiarity scores mean low distance
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}

    #for performance reasons transform matrix into nested dict
    similarity_scores_of_activities = get_nested_dict(similarity_scores_of_activities)

    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, similarity_scores_of_activities)
    triplet_value_for_class = list()
    for replaced_activity in replaced_activities:
        out_of_class_activities = get_out_of_class_activities(similarity_scores_of_activities, replaced_activity)
        in_class_combinations = get_in_class_combinations(replaced_activities_dict[replaced_activity])
        triplet_value_for_pairs = list()
        for replaced_activity_pair in in_class_combinations:
            triplet_value_for_pair = list()
            for out_of_class_activity in out_of_class_activities:
                triplet_value_for_pair.append(triplet_is_in_class(replaced_activity_pair[0], replaced_activity_pair[1], out_of_class_activity, similarity_scores_of_activities))
            triplet_value_for_pairs.append(sum(triplet_value_for_pair) / len(triplet_value_for_pair))
        triplet_value_for_class.append(sum(triplet_value_for_pairs) / len(triplet_value_for_pairs))
    triplet_value_for_log = sum(triplet_value_for_class) / len(triplet_value_for_class)
    return triplet_value_for_log


def triplet_is_in_class(activity_1_in_class, activity_2_in_class, activity_3_out_of_class, activity_distance_matrix):
    if activity_distance_matrix[activity_1_in_class][activity_2_in_class] < activity_distance_matrix[activity_1_in_class][activity_3_out_of_class]:
        return 1
    else: return 0




def get_nested_dict(similarity_scores_of_activities):
    # Initialize the nested dictionary
    nested_dict = {}

    for (outer_key, inner_key), value in similarity_scores_of_activities.items():
        # Ensure the outer key exists in the nested dictionary
        if outer_key not in nested_dict:
            nested_dict[outer_key] = {}
        # Assign the value to the inner key
        nested_dict[outer_key][inner_key] = value
    return nested_dict


def get_replaced_activities_dict(replaced_activities, activity_distance_matrix):
    replaced_activities_dict = {key: list() for key in replaced_activities}
    for activity in activity_distance_matrix:
        replaced_activity_with_split = activity.split(':')
        if len(replaced_activity_with_split) > 1:
            if not replaced_activity_with_split[1].isdigit():
                sys.exit("Naming Error")
            else:
                replaced_activities_dict[replaced_activity_with_split[0]].append(activity)
    return replaced_activities_dict

def get_out_of_class_activities(activity_distance_matrix, replaced_activity):
    out_of_class_activities = list()
    for activity in activity_distance_matrix.keys():
        if activity.split(':')[0] != replaced_activity:
            out_of_class_activities.append(activity)
    return out_of_class_activities

def get_in_class_combinations(replaced_activities_of_class):
    in_class_combinations = list(combinations(replaced_activities_of_class, 2))
    result = []
    for comb in in_class_combinations:
        result.append(comb)  # Add original combination
        result.append(comb[::-1])  # Add reversed combination
    return result


def get_diameter(activity_distance_matrix_dict,
                 activities_to_replace_with_count, reverse, alphabet):
    diameter_list = list()
    for activity_distance_function in activity_distance_matrix_dict:
        for replaced_activities in activity_distance_matrix_dict[activity_distance_function].keys():
            avg_triplet_value = get_avg_diameter_value(replaced_activities, activity_distance_matrix_dict[activity_distance_function][
                                                            replaced_activities], activities_to_replace_with_count, reverse, alphabet)
            diameter_list.append(avg_triplet_value)

    return sum(diameter_list) / len(diameter_list)


def get_avg_diameter_value(replaced_activities, similarity_scores_of_activities, activities_to_replace_with_count,
                            reverse, alphabet):
    #high bose similiarity scores mean low distance
    if reverse:
        similarity_scores_of_activities = {key: -value for key, value in similarity_scores_of_activities.items()}

    normalized_similarity_scores_of_activities = get_normalized_similarity_scores_of_activities(similarity_scores_of_activities)

    #for performance reasons transform matrix into nested dict
    normalized_similarity_scores_of_activities = get_nested_dict(normalized_similarity_scores_of_activities)
    replaced_activities_dict = get_replaced_activities_dict(replaced_activities, normalized_similarity_scores_of_activities)
    diameter_per_class = list()
    for replaced_activity in replaced_activities:
        in_class_combinations = get_in_class_combinations(replaced_activities_dict[replaced_activity])
        diameter_in_class = list()
        for in_class_combination in in_class_combinations:
            diameter_in_class.append(normalized_similarity_scores_of_activities[in_class_combination[0]][in_class_combination[1]])
        diameter_per_class.append(sum(diameter_in_class) / len(diameter_in_class))
    avg_diameter_for_log = sum(diameter_per_class) / len(diameter_per_class)
    return avg_diameter_for_log

def get_normalized_similarity_scores_of_activities(similarity_scores_of_activities):
    min_value = min(similarity_scores_of_activities.values())
    max_value = max(similarity_scores_of_activities.values())

    # Normalize the values
    normalized_data = {
        key: (value - min_value) / (max_value - min_value)
        for key, value in similarity_scores_of_activities.items()
    }

    return normalized_data
