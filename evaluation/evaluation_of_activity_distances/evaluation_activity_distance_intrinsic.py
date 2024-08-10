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

from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict

from evaluation.data_util.util_activity_distances_intrinsic import (
    get_log_control_flow_perspective, get_activities_to_replace,
    get_logs_with_replaced_activities_dict,
    get_knn_dict, get_precision_at_k
)

def evaluate_intrinsic(activity_distance_functions, log_list):
    for log_name in log_list:
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
        #pm4py.view_process_tree(pm4py.discover_process_tree_inductive(log))
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        alphabet = get_alphabet(log_control_flow_perspective)

        for activity_distance_function in activity_distance_functions:

            ########################
            # Intrinsic evaluation #
            ########################

            combinations = [
                (
                different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, [activity_distance_function])
                for different_activities_to_replace_count in range(1, len(alphabet))
                for activities_to_replace_with_count in range(2, 20+1)
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

    # Clean up to save memory
    del logs_with_replaced_activities_dict
    gc.collect()


    if "Bose 2009 Substitution Scores" == activity_distance_function[0]:
        reverse=True #high values = high similarity
    else:
        reverse=False #high values = high distances

    # 4: evaluation of all activity distance matrices
    w_minus_one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, activities_to_replace_with_count-1)
    precision_at_w_minus_1_dict = get_precision_at_k(w_minus_one_nn_dict, activity_distance_function)
    print(precision_at_w_minus_1_dict)
    precision_at_w_minus_1 = precision_at_w_minus_1_dict[activity_distance_function[0]]

    one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, 1)
    precision_at_1_dict = get_precision_at_k(one_nn_dict, activity_distance_function)
    precision_at_1 = precision_at_1_dict[activity_distance_function[0]]

    #precision = precision_at_k_dict["De Koninck 2018 act2vec"]

    return different_activities_to_replace_count, activities_to_replace_with_count, precision_at_w_minus_1, precision_at_1

def visualization_intrinsic_evaluation(results, activity_distance_function, log_name):
    # Create DataFrame from results
    df = pd.DataFrame(results, columns=['r', 'w', 'precision@w-1', 'precision@1'])

    #heat map precision@w-1
    result = df.pivot(index='w', columns='r', values='precision@w-1')
    # Plotting
    rc('font', **{'family': 'serif', 'size': 20})
    f, ax = plt.subplots(figsize=(11, 15))
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, annot=True, linewidth=.5)
    ax.invert_yaxis()
    ax.set_title("precision@w-1 for " + log_name + "\n" +activity_distance_function, pad=20)
    plt.savefig("Histo.pdf", format="pdf", transparent=True)
    plt.show()
    #heat map precision@1
    result = df.pivot(index='w', columns='r', values='precision@1')
    # Plotting
    rc('font', **{'family': 'serif', 'size': 20})
    f, ax = plt.subplots(figsize=(11, 15))
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, annot=True, linewidth=.5)
    ax.invert_yaxis()
    ax.set_title("precision@1 for " + log_name + "\n" +activity_distance_function, pad=20)
    plt.savefig("Histo.pdf", format="pdf", transparent=True)
    plt.show()




if __name__ == '__main__':

    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    #activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    ##############################################################################

    ##############################################################################
    # intrinsic - event logs we want to evaluate
    log_list = list()
    #log_list.append("Sepsis")
    log_list.append("repairExample")
    ##############################################################################

    ##############################################################################
    # intrinsic - event logs we want to evaluate
    evluation_measure_list = list()
    #evluation_measure_list.append("precision@w-1")
    #evluation_measure_list.append("precision@1")
    ##############################################################################


    evaluate_intrinsic(activity_distance_functions, log_list)









