import gc
from multiprocessing import Pool

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log import get as variants_module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances_intrinsic import (
    get_log_control_flow_perspective, get_alphabet, get_activities_to_replace,
    get_logs_with_replaced_activities_dict, get_activity_distance_matrix_dict,
    get_knn_dict, get_precision_at_k
)
from evaluation.data_util.util_activity_distances_extrensic import (
    get_sublog_list, get_activity_distance_matrix
)

def evaluate_intrinsic(activity_distance_functions, log_list):
    for log_name in log_list:
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
        pm4py.view_process_tree(pm4py.discover_process_tree_inductive(log))
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        alphabet = get_alphabet(log_control_flow_perspective)


        for activity_distance_function in activity_distance_functions:

            ########################
            # Intrinsic evaluation #
            ########################

            combinations = [
                (
                different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, [activity_distance_function])
                for different_activities_to_replace_count in range(1, 6)
                for activities_to_replace_with_count in range(2, 6)
            ]

            with Pool() as pool:
                results = pool.map(intrinsic_evaluation, combinations)

            visualization_intrinsic_evaluation(results, activity_distance_function, log_name)



def intrinsic_evaluation(args):
    different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, activity_distance_function = args
    # 1: get the activities that we want to replace in each run
    activities_to_replace_in_each_run_list = get_activities_to_replace(alphabet, different_activities_to_replace_count)

    # 2: replace activities
    logs_with_replaced_activities_dict = get_logs_with_replaced_activities_dict(
        activities_to_replace_in_each_run_list, log_control_flow_perspective,
        different_activities_to_replace_count, activities_to_replace_with_count
    )

    # 3: compute for all logs all activity distance matrices
    n_gram_size_bose_2009 = 3

    activity_distance_matrix_dict = get_activity_distance_matrix_dict(
        activity_distance_function, logs_with_replaced_activities_dict, n_gram_size_bose_2009
    )

    # 4: evaluation of all activity distance matrices
    knn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count)
    precision_at_k_dict = get_precision_at_k(knn_dict, activity_distance_function)

    print(precision_at_k_dict)

    precision = precision_at_k_dict[activity_distance_function[0]]
    #precision = precision_at_k_dict["De Koninck 2018 act2vec"]

    # Clean up to save memory
    del logs_with_replaced_activities_dict
    gc.collect()

    return different_activities_to_replace_count, activities_to_replace_with_count, precision

def visualization_intrinsic_evaluation(results, activity_distance_function, log_name):
    # Create DataFrame from results
    df = pd.DataFrame(results, columns=['r', 'w', 'precision@'])
    result = df.pivot(index='w', columns='r', values='precision@')

    # Plotting
    rc('font', **{'family': 'serif', 'size': 20})
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, annot=True, linewidth=.5)
    ax.invert_yaxis()
    ax.set_title(activity_distance_function + ' - ' + log_name, pad=20)
    plt.savefig("Histo.pdf", format="pdf", transparent=True)
    plt.show()

def evaluate_extrensic(activity_distance_functions, event_log_folder):

    sublog_list = get_sublog_list(event_log_folder)

    log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]

    alphabet = get_alphabet(log_control_flow_perspective)

    combinations = [
        (log_control_flow_perspective, activity_distance_function, alphabet)
        for activity_distance_function in activity_distance_functions
    ]

    with Pool() as pool:
        activity_distance_matrix_dict_list = pool.map(get_activity_distance_matrix_dict_list, combinations)

    combinations = [
        (sublog_list, activity_distance_matrix_dict, alphabet)
        for activity_distance_matrix_dict in activity_distance_matrix_dict_list
    ]
    with Pool() as pool:
        results = pool.map(get_activity_distance_matrix_dict_list, combinations)



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




if __name__ == '__main__':

    ##############################################################################
    # activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    # activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    ##############################################################################

    ##############################################################################
    # intrinsic - event logs we want to evaluate
    log_list = list()
    #log_list.append("Sepsis")
    log_list.append("repairExample")
    ##############################################################################

    #evaluate_intrinsic(activity_distance_functions, log_list)

    ##############################################################################
    # extrensic - event logs we want to evaluate
    event_log_folder = "pdc_2022"
    #log_list.append("Sepsis")
    ##############################################################################

    evaluate_extrensic(activity_distance_functions, event_log_folder)








