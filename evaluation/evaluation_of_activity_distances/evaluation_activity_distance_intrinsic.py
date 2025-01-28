import multiprocessing
import random
from multiprocessing import Pool

from pm4py.objects.log.importer.xes import importer as xes_importer

from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict, \
    get_log_control_flow_perspective_with_short_activity_names
from evaluation.data_util.util_activity_distances_extrinsic import get_sublog_list
from evaluation.data_util.util_activity_distances_intrinsic import (
    get_log_control_flow_perspective, get_activities_to_replace,
    get_logs_with_replaced_activities_dict,
    get_knn_dict, get_precision_at_k, save_intrinsic_results, get_triplet, get_diameter
)


def evaluate_intrinsic(activity_distance_functions, log_list, r_min, w, sampling_size):
    # Load event logs from files
    for log_name in log_list:
        # If multiple XES files need to be imported for one log, load all files from the specified folder.
        if log_name[:4] == "bpic" or log_name[:3] == "pdc":
            sublog_list = get_sublog_list(log_name)
            log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]
        else:
            log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
            log_control_flow_perspective = get_log_control_flow_perspective(log)

        alphabet = get_alphabet(log_control_flow_perspective)
        # Transform activity labels into sequential numbers to improve performance.
        #log_control_flow_perspective = get_log_control_flow_perspective_with_short_activity_names(
        #    log_control_flow_perspective, alphabet)
        #alphabet = get_alphabet(log_control_flow_perspective)


        # Limit the number of activity replacements to 'r_min' when the total number of activities in the log exceeds 'r_min' to enhance performance.
        r = min(r_min, len(alphabet))

        # Define all subproblems required for our intrinsic evaluation, allowing each subproblem to be assigned to a separate core for parallel processing.
        combinations = [
            (
                different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective,
                alphabet, activity_distance_functions, sampling_size)
            for different_activities_to_replace_count in range(1, r + 1)
            for activities_to_replace_with_count in range(2, w + 1)
        ]

        # Limit used cores, to ensure system responsiveness
        total_cores = multiprocessing.cpu_count()

        # Calculate 75% of the available cores
        cores_to_use = int(total_cores * 0.75)

        # Ensure at least one core is used
        cores_to_use = max(1, cores_to_use)

        # Start the computation
        with Pool(processes=cores_to_use) as pool:
            results = pool.map(intrinsic_evaluation, combinations)

        # Combine and save results
        save_intrinsic_results(activity_distance_functions, results, log_name, r, w, sampling_size)


def intrinsic_evaluation(args):
    different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet, activity_distance_function_list, sampling_size = args

    # 1: get the activities that we want to replace in each run
    activities_to_replace_in_each_run_list = get_activities_to_replace(alphabet, different_activities_to_replace_count,
                                                                       sampling_size)

    print("start ---- r:" + str(different_activities_to_replace_count) + " w: " + str(activities_to_replace_with_count))

    # 1.1: limit the number of logs for better performance
    if len(activities_to_replace_in_each_run_list) >= sampling_size:
        activities_to_replace_in_each_run_list = random.sample(list(activities_to_replace_in_each_run_list),
                                                               sampling_size)
        set(activities_to_replace_in_each_run_list)

    results_list = list()

    for activity_distance_function in activity_distance_function_list:
        activity_distance_function = [activity_distance_function]

        diameter_list = list()
        precision_at_w_minus_1_list = list()
        precision_at_1_list = list()
        triplet_list = list()

        for activities_to_replace in activities_to_replace_in_each_run_list:
            # 2: replace activities
            logs_with_replaced_activities_dict = get_logs_with_replaced_activities_dict(
                [activities_to_replace], log_control_flow_perspective,
                different_activities_to_replace_count, activities_to_replace_with_count
            )
            n_gram_size_bose_2009 = 3


            # 3: Compute distances between activities
            activity_distance_matrix_dict = get_activity_distance_matrix_dict(
                activity_distance_function, logs_with_replaced_activities_dict, n_gram_size_bose_2009
            )

            if "Bose 2009 Substitution Scores" == activity_distance_function[0]:
                reverse = True  # high values = high similarity
            else:
                reverse = False  # high values = high distances

            # 4: evaluation of all activity distance matrices
            #if activities_to_replace_with_count == 11:
            #    print(2)


            #Average Diameter Distance
            diameter_list.append(get_diameter(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet))

            #Precision@w-1
            w_minus_one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse,
                                               activities_to_replace_with_count - 1)
            precision_at_w_minus_1_dict = get_precision_at_k(w_minus_one_nn_dict, activity_distance_function)
            precision_at_w_minus_1_list.append(precision_at_w_minus_1_dict[activity_distance_function[0]])

            #Nearest Neighbor
            one_nn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, 1)
            precision_at_1_dict = get_precision_at_k(one_nn_dict, activity_distance_function)
            precision_at_1_list.append(precision_at_1_dict[activity_distance_function[0]])

            #Triplet
            triplet_list.append(get_triplet(activity_distance_matrix_dict, activities_to_replace_with_count, reverse, alphabet))

            




        diameter = sum(diameter_list) / len(diameter_list)
        precision_at_w_minus_1 = sum(precision_at_w_minus_1_list) / len(precision_at_w_minus_1_list)
        precision_at_1 = sum(precision_at_1_list) / len(precision_at_1_list)
        triplet = sum(triplet_list) / len(triplet_list)
        results_list.append((different_activities_to_replace_count, activities_to_replace_with_count, diameter,
                             precision_at_w_minus_1, precision_at_1, triplet))

    print("end ---- r:" + str(different_activities_to_replace_count) + " w: " + str(
        activities_to_replace_with_count) + " sampling size: " + str(sampling_size))
    return results_list


if __name__ == '__main__':
    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    #activity_distance_functions.append("Bose 2009 Substitution Scores")
    #activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    activity_distance_functions.append("Chiorrini 2023 Embedding Process Structure")
    #activity_distance_functions.append("Unit Distance")
    # activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")

    ##############################################################################
    r_min = 3
    w = 3
    sampling_size = 1
    print(sampling_size)
    ##############################################################################
    # intrinsic - event logs we want to evaluate
    log_list = list()
    #log_list.append("Sepsis")
    log_list.append("repairExample")
    #log_list.append("bpic_2015")
    #log_list.append("Sepsis")
    #log_list.append("Road Traffic Fine Management Process")
    # log_list.append("bpic_2015")
    # log_list.append("PDC 2016")
    #log_list.append("BPI Challenge 2015 1")
    # log_list.append("pdc_2022")
    # log_list.append("PDC 2017")
    #log_list.append("PDC 2019")
    #log_list.append("BPI Challenge 2017")

    # log_list.append("BPI Challenge 2017")
    #log_list.append("WABO")

    print(log_list)

    # log_list.append("2019_1")
    ##############################################################################

    ##############################################################################
    # intrinsic - event logs we want to evaluate
    evluation_measure_list = list()
    # evluation_measure_list.append("precision@w-1")
    # evluation_measure_list.append("precision@1")
    ##############################################################################

    evaluate_intrinsic(activity_distance_functions, log_list, r_min, w, sampling_size)
