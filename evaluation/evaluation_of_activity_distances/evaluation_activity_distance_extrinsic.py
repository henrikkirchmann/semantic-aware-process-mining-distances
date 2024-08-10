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

from evaluation.data_util.util_activity_distances import get_alphabet, get_activity_distance_matrix_dict_list

from evaluation.data_util.util_activity_distances_extrinsic import (
    get_sublog_list, get_trace_distances, get_precision_values, get_log_with_trace_ids
)

def evaluate_extrinsic(activity_distance_functions, event_log_folder):

    sublog_list = get_sublog_list(event_log_folder)

    sublogsize_list = [len(sublog) for sublog in sublog_list]

    log_control_flow_perspective = [inner for outer in sublog_list for inner in outer]

    alphabet = get_alphabet(log_control_flow_perspective)

    combinations = [
        (log_control_flow_perspective, activity_distance_function, alphabet)
        for activity_distance_function in activity_distance_functions
    ]

    with Pool() as pool:
        activity_distance_matrix_dict_list = pool.map(get_activity_distance_matrix_dict_list, combinations)

    trace_sublog_list_all_list, trace_sublog_list_all_list_flat = get_log_with_trace_ids(log_control_flow_perspective, sublogsize_list)

    combinations = [
        (sublog, trace_sublog_list_all_list_flat, activity_distance_matrix_dict, alphabet, sublogsize_list)
        for activity_distance_matrix_dict in activity_distance_matrix_dict_list
        for sublog in trace_sublog_list_all_list
    ]

    with Pool() as pool:
        results = pool.map(extrinisc_evaluation, combinations)

    print("a")




def extrinisc_evaluation(args):
    trace_list, all_trace_list, activity_distance_matrix_dict, alphabet, sublogsize_list = args

    precison_list = list()

    for trace in trace_list: #[:10]
        trace_distance_list = get_trace_distances(trace, all_trace_list, activity_distance_matrix_dict)

        precison_list.append(get_precision_values(trace_distance_list, trace, sublogsize_list))

    nn = 0
    pre = 0
    for precion_values in precison_list:
        nn += precion_values[0]
        pre += precion_values[1]
    nn = nn / len(precison_list)
    pre = pre / len(precison_list)



    #avg = sum(precison_list) / len(precison_list)
    print(next(iter(activity_distance_matrix_dict)) + " NN: "+ str(nn) + " Pre: " + str(pre))
    return precison_list


if __name__ == '__main__':

    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    #activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    ##############################################################################

    ##############################################################################
    # extrensic - event logs we want to evaluate
    event_log_folder = "pdc_2019"
    #log_list.append("Sepsis")
    ##############################################################################

    evaluate_extrinsic(activity_distance_functions, event_log_folder)
