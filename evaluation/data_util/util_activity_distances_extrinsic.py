import os

from collections import defaultdict

import pm4py

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from definitions import ROOT_DIR
from pm4py.objects.log.importer.xes import importer as xes_importer


def get_sublog_list(folder):
    i = 0
    sublog_list = list()
    for sublog_name in os.listdir(ROOT_DIR + '/event_logs/' + folder):
        sublog = xes_importer.apply(ROOT_DIR + '/event_logs/' + folder + "/" + sublog_name)
        #pt = pm4py.discover_process_tree_inductive(sublog)
        #pm4py.view_process_tree(pt)
        sublog_control_flow_perspective = get_log_control_flow_perspective(sublog)
        sublog_list.append(sublog_control_flow_perspective)
        i += 1
        if i == 10:
            break
    return sublog_list



#get_sublog_list("pdc_2019")