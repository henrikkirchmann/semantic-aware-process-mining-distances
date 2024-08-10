import os

from collections import defaultdict

import pm4py

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from definitions import ROOT_DIR
from pm4py.objects.log.importer.xes import importer as xes_importer
from distances.trace_distances.edit_distance.levenshtein.algorithm import get_levenshtein_distance

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

def get_trace_distances(trace, all_trace_list, activity_distance_matrix_dict):
    trace_distance_list = list()
    for trace_tuple in all_trace_list:
        if trace[0] != trace_tuple[0]:  # do not compute distance between same trace
            trace_distance_list.append((get_levenshtein_distance(trace[2], trace_tuple[2], activity_distance_matrix_dict),) + trace_tuple)
    return trace_distance_list
#get_sublog_list("pdc_2019")

def get_log_with_trace_ids(log_control_flow_perspective, sublogsize_list):
    trace_sublog_pair_list = list()
    trace_sublog_list_all_list = list()
    trace_sublog_list_all_list_flat = list()
    i = 0
    sublog_id = 0
    trace_id = 0
    for trace in log_control_flow_perspective:
        trace_sublog_pair_list.append((trace_id, sublog_id, trace))
        if i == sublogsize_list[sublog_id]-1:
            sublog_id += 1
            trace_sublog_list_all_list.append(trace_sublog_pair_list)
            trace_sublog_list_all_list_flat.extend(trace_sublog_pair_list)
            trace_sublog_pair_list = list()
            i = 0
        else:
            i += 1
        trace_id += 1
    return trace_sublog_list_all_list, trace_sublog_list_all_list_flat


def get_precision_values(trace_distance_list, trace, sublogsize_list):
    # Sort distances by the similarity score (distance)
    trace_distance_list.sort(key=lambda x: x[0], reverse=False)

    # precision@1 (nn)
    if trace_distance_list[0][2] == trace[1]:
        precision_at_1 = 1
    else:
        precision_at_1 = 0

    k = sublogsize_list[trace[1]]
    # precision_at_k
    same_sub_log_traces_count = 0
    for trace_tuple in trace_distance_list[:k]:
        if trace_tuple[2] == trace[1]:
            same_sub_log_traces_count += 1
    precision_at_k = same_sub_log_traces_count / k

    return precision_at_1, precision_at_k