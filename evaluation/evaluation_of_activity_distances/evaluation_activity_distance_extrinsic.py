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
    get_sublog_list, get_activity_distance_matrix
)

def evaluate_extrensic(activity_distance_functions, event_log_folder):

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

    combinations = [
        (sublog_list, activity_distance_matrix_dict, alphabet)
        for activity_distance_matrix_dict in activity_distance_matrix_dict_list
    ]
    with Pool() as pool:
        results = pool.map(get_activity_distance_matrix_dict_list, combinations)

if __name__ == '__main__':

    ##############################################################################
    # intrinsic - activity_distance_functions we want to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    # activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    #activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    ##############################################################################

    ##############################################################################
    # extrensic - event logs we want to evaluate
    event_log_folder = "pdc_2022"
    #log_list.append("Sepsis")
    ##############################################################################

    evaluate_extrensic(activity_distance_functions, event_log_folder)

