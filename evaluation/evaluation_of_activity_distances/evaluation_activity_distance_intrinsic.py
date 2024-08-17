import gc
import multiprocessing
import sys
from multiprocessing import Pool
from pathlib import Path
import random
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log import get as variants_module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from definitions import ROOT_DIR
import time



from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict, get_log_control_flow_perspective_with_short_activity_names, get_obj_size, unresponsiveness_prediction

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
        #print(get_obj_size(log_control_flow_perspective))
        log_control_flow_perspective = get_log_control_flow_perspective_with_short_activity_names(log_control_flow_perspective, alphabet)
        print(get_obj_size(log_control_flow_perspective))
        #active
        alphabet = get_alphabet(log_control_flow_perspective)
        #for activity_distance_function in activity_distance_functions:

        ########################
        # Intrinsic evaluation #
        ########################

        r = len(alphabet)
        w = 20
        sampling_size = None

        if unresponsiveness_prediction(get_obj_size(log_control_flow_perspective), len(alphabet), r, w):
            #set sampling size to high as possible, but enough room for not too much ram consumption take
            step_size = 1
            max_value = 100000
            for sampling_size_test in range(1, max_value + 1, step_size):
                if not unresponsiveness_prediction(get_obj_size(log_control_flow_perspective), len(alphabet), r, w,
                                                sampling_size_test):
                    sampling_size = sampling_size_test
                else:
                    if sampling_size is None:
                        sampling_size = 1
                        print("System might run out of memory.")
                    break

        print(sampling_size)


        combinations = [
            (
            different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, activity_distance_functions, sampling_size)
            for different_activities_to_replace_count in range(1, r+1)
            for activities_to_replace_with_count in range(2, w+1)
        ]

        # limit used cores, for system responsiveness
        total_cores = multiprocessing.cpu_count()

        # Calculate 75% of the available cores
        cores_to_use = int(total_cores * 0.75)

        # Ensure at least one core is used
        cores_to_use = max(1, cores_to_use)

        with Pool(processes=cores_to_use) as pool:
            results = pool.map(intrinsic_evaluation, combinations)

    activity_distance_function_index = 0
    for activity_distance_function in activity_distance_functions:
        results_per_activity_distance_function = list()
        for result in results:
            #if result[activity_distance_function_index][4] == activity_distance_function:
            results_per_activity_distance_function.append(result[activity_distance_function_index])
        visualization_intrinsic_evaluation(results_per_activity_distance_function, activity_distance_function, log_name, r, w)
        activity_distance_function_index = activity_distance_function_index + 1


def intrinsic_evaluation(args):
    different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, activity_distance_function_list, sampling_size = args
    # 1: get the activities that we want to replace in each run
    activities_to_replace_in_each_run_list = get_activities_to_replace(alphabet, different_activities_to_replace_count)
    #print("start ---- r:" + str(different_activities_to_replace_count) + " w: "+str(activities_to_replace_with_count))
    #1.1: limit the number of logs for performance
    #max_number_of_logs = 15
    if len(activities_to_replace_in_each_run_list) > sampling_size:
        activities_to_replace_in_each_run_list = random.sample(activities_to_replace_in_each_run_list, sampling_size)

    # 2: replace activities
    start_time = time.time()
    logs_with_replaced_activities_dict = get_logs_with_replaced_activities_dict(
        activities_to_replace_in_each_run_list, log_control_flow_perspective,
        different_activities_to_replace_count, activities_to_replace_with_count
    )
    del logs_with_replaced_activities_dict
    gc.collect()
    #print(str((time.time() - start_time)) +" seconds ---" + " r:" + str(different_activities_to_replace_count) + " w: "+str(activities_to_replace_with_count))
    #return list()
    # 3: compute for all logs all activity distance matrices
    n_gram_size_bose_2009 = 3

    results_list = list()

    for activity_distance_function in activity_distance_function_list:
        activity_distance_function = [activity_distance_function]
        activity_distance_matrix_dict = get_activity_distance_matrix_dict(
            activity_distance_function, logs_with_replaced_activities_dict, n_gram_size_bose_2009
        )

        # Clean up to save memory
        #del logs_with_replaced_activities_dict
        #gc.collect()


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

        results_list.append((different_activities_to_replace_count, activities_to_replace_with_count, precision_at_w_minus_1, precision_at_1))

    #precision = precision_at_k_dict["De Koninck 2018 act2vec"]
    print("end ---- r:" + str(different_activities_to_replace_count + " w: "+str(activities_to_replace_with_count)))


    return results_list

def visualization_intrinsic_evaluation(results, activity_distance_function, log_name, r, w):
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
    Path(ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k").mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k/" + "pre_" + activity_distance_function + "_" + log_name + "_r:" + str(r) + "_w:" + str(w) + ".pdf", format="pdf", transparent=True)
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
    Path(ROOT_DIR + "/results/activity_distances/intrinsic/nn").mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT_DIR + "/results/activity_distances/intrinsic/nn/" + "nn" + activity_distance_function + "_" + log_name + "_r:" + str(r) + "_w:" + str(w) + ".pdf", format="pdf", transparent=True)
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
    log_list.append("repairExample")
    #log_list.append("Sepsis")
    #log_list.append("Road_Traffic_Fine_Management_Process")
    ##############################################################################

    ##############################################################################
    # intrinsic - event logs we want to evaluate
    evluation_measure_list = list()
    #evluation_measure_list.append("precision@w-1")
    #evluation_measure_list.append("precision@1")
    ##############################################################################


    evaluate_intrinsic(activity_distance_functions, log_list)








