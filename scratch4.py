import multiprocessing
from multiprocessing import Pool

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict_list, print_log_stats
from evaluation.data_util.util_activity_distances_extrinsic import get_sublog_list, get_log_with_trace_ids, \
    get_activity_clustering

# Sample data: replace with your list of lists of lists

# Function to flatten lists of lists
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

# Calculate the percentage of similarity
def similarity_percentage(list1, list2):
    number_of_same_traces = 0
    matched_indices = set()  # Track indices of matched traces in list2

    for trace1 in list1:
        for idx, trace2 in enumerate(list2):
            if idx in matched_indices:
                continue  # Skip if trace2 at idx is already matched

            if len(trace1) != len(trace2):
                continue  # Skip if lengths differ

            if all(trace1[i] == trace2[i] for i in range(len(trace1))):  # Check full match
                number_of_same_traces += 1
                matched_indices.add(idx)  # Mark trace2 at idx as matched
                break  # Move to the next trace1 after finding a match

    return number_of_same_traces / len(list1) if list1 else 0


def replace_activities_with_clusters(trace, activity_clustering):
    """
    Replace activities in a trace with their corresponding cluster values from activity_clustering.
    If an activity has a value of -1 in activity_clustering, keep the activity as it is.

    Args:
        trace (list of str): List of activities.
        activity_clustering (dict): Dictionary mapping activities to clusters.

    Returns:
        list: A new trace with activities replaced by cluster values where applicable.
    """
    newTrace = []
    for activity in trace:
        newActivity = activity_clustering[activity]
        if newActivity == -1:
            newActivity = activity
        newTrace.append(newActivity)
    return newTrace


def log_similarity(log_name, activity_distance_functions):

    if log_name[:4] == "bpic" or log_name[:3] == "pdc":
        sublog_list = get_sublog_list(log_name)
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]

    alphabet = get_alphabet(log_control_flow_perspective)

    print_log_stats(log_control_flow_perspective, alphabet)

    combinations = [
        (log_control_flow_perspective, activity_distance_function, alphabet)
        for activity_distance_function in activity_distance_functions
    ]

    # limit used cores, for system responsiveness
    total_cores = multiprocessing.cpu_count()

    # Calculate 75% of the available cores
    cores_to_use = int(total_cores * 0.80)

    # Ensure at least one core is used
    cores_to_use = max(1, cores_to_use)

    with Pool(processes=cores_to_use) as pool:
        activity_distance_matrix_dict_list = pool.map(get_activity_distance_matrix_dict_list, combinations)

    print("Distances Computed")

    trace_sublog_list_all_list, trace_sublog_list_all_list_flat = get_log_with_trace_ids(log_control_flow_perspective,
                                                                                         sublogsize_list)
    #change similiarities to distances
    for admd in activity_distance_matrix_dict_list:
        if "Bose 2009 Substitution Scores" in admd:
            for key in admd["Bose 2009 Substitution Scores"].keys():
                admd["Bose 2009 Substitution Scores"][key] = 1 - admd[
                    "Bose 2009 Substitution Scores"][key]

    activity_clustering = get_activity_clustering(activity_distance_matrix_dict_list,  )


    for log in sublog_list:
        for trace in log:
            trace = replace_activities_with_clusters(trace, activity_clustering)


    # Create an empty matrix to store similarities
    n = len(sublog_list)
    similarity_matrix = np.zeros((n, n))

    # Fill the matrix with similarity values
    for i in range(n):
        print(i)
        for j in range(n):
            similarity_matrix[i, j] = similarity_percentage(sublog_list[i], sublog_list[j])

    # Plot the heatmap
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".1f",
                xticklabels=[f'List {i+1}' for i in range(n)],
                yticklabels=[f'List {i+1}' for i in range(n)])
    plt.title('Pairwise Similarity Percentage Heatmap')
    plt.show()

if __name__ == '__main__':
    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    # activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    ##############################################################################

    ##############################################################################
    # extrensic - event logs we want to evaluate
    #event_log_folder = "bpic_2015"
    #event_log_folder = "PDC 2016"
    #event_log_folder = "pdc_2020"
    event_log_folder = "pdc_2019"
    #event_log_folder = "pdc_2020"
    #event_log_folder = "pdc_2020"
    #event_log_folder = "pdc_2022"
    #event_log_folder = "pdc_2023"
    #event_log_folder = "WABO"
    #event_log_folder = "pdc_2021"
    #event_log_folder = "pdc_2023"
    #event_log_folder = "PDC 2017"
    #event_log_folder = "repairExample"
    log_similarity(event_log_folder, activity_distance_functions)


