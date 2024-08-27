import multiprocessing
from multiprocessing import Pool

from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict_list, get_obj_size, print_log_stats
from evaluation.data_util.util_activity_distances_extrinsic import (
    get_sublog_list, get_trace_distances, get_precision_values, get_log_with_trace_ids, print_avg_values, get_sampled_sublogs
)

from definitions import ROOT_DIR
from pm4py.objects.log.importer.xes import importer as xes_importer
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective

def evaluate_extrinsic(activity_distance_functions, log_name):

    #for log_name in event_log_folder:
    if log_name[:4] == "bpic" or log_name[:3] == "pdc":
        sublog_list = get_sublog_list(log_name)
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]
    else:

        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
        #pm4py.view_process_tree(pm4py.discover_process_tree_inductive(log))
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        sublog_list = [log]
        sublogsize_list = [len(sublog) for sublog in sublog_list]
        #print(get_obj_size(log_control_flow_perspective))
    alphabet = get_alphabet(log_control_flow_perspective)
    #log_control_flow_perspective = get_log_control_flow_perspective_with_short_activity_names(
        #log_control_flow_perspective, alphabet)
    print(get_obj_size(log_control_flow_perspective))

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

    #sample sublogs
    percentage = 0.01
    print(percentage)
    trace_sublog_list_all_list = get_sampled_sublogs(trace_sublog_list_all_list, percentage)

    combinations = [
        (sublog, trace_sublog_list_all_list_flat, activity_distance_matrix_dict, alphabet, sublogsize_list)
        for activity_distance_matrix_dict in activity_distance_matrix_dict_list
        for sublog in trace_sublog_list_all_list
    ]

    with Pool(processes=cores_to_use) as pool:
        results = pool.map(extrinisc_evaluation, combinations)

    print_avg_values(results, activity_distance_functions)


def extrinisc_evaluation(args):
    trace_list, all_trace_list, activity_distance_matrix_dict, alphabet, sublogsize_list = args

    precison_list = list()

    for trace in trace_list:
        trace_distance_list = get_trace_distances(trace, all_trace_list, activity_distance_matrix_dict)

        precison_list.append(get_precision_values(trace_distance_list, trace, sublogsize_list))

    nn = 0
    pre = 0
    pre10 = 0
    for precion_values in precison_list:
        nn += precion_values[0]
        pre += precion_values[1]
        pre10 += precion_values[2]
    nn = nn / len(precison_list)
    pre = pre / len(precison_list)
    pre10 = pre10 / len(precison_list)

    # avg = sum(precison_list) / len(precison_list)
    print(next(iter(activity_distance_matrix_dict)) + " NN: " + str(nn) + " Pre: " + str(pre) + " Pre10: " + str(pre10))
    return (next(iter(activity_distance_matrix_dict)), nn, pre, pre10)


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
    #event_log_folder = "pdc_2016"
    #event_log_folder = "pdc_2020"
    #event_log_folder = "pdc_2019"
    #event_log_folder = "pdc_2020"
    #event_log_folder = "pdc_2020"
    #event_log_folder = "pdc_2022"
    #event_log_folder = "pdc_2023"
    #event_log_folder = "wabo_all"
    event_log_folder = "pdc_2021"
    #event_log_folder = "pdc_2023"
    #event_log_folder = "pdc_2017"




    print(event_log_folder)


    #event_log_folder = "repairExample"
    #event
    # log_list.append("Sepsis")
    ##############################################################################

    evaluate_extrinsic(activity_distance_functions, event_log_folder)
