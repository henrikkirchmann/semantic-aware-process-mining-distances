import gc
from multiprocessing import Pool
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.variants.log import get as variants_module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from definitions import ROOT_DIR
from evaluation.data_util.algorithm import (
    get_log_control_flow_perspective, get_alphabet, get_activities_to_replace,
    get_logs_with_replaced_activities_dict, get_activity_distance_matrix_dict,
    get_knn_dict, get_precision_at_k
)


def process_combination(args):
    different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet = args
    # 1: get the activities that we want to replace in each run
    activities_to_replace_in_each_run_list = get_activities_to_replace(alphabet, different_activities_to_replace_count)

    # 2: replace activities
    logs_with_replaced_activities_dict = get_logs_with_replaced_activities_dict(
        activities_to_replace_in_each_run_list, log_control_flow_perspective,
        different_activities_to_replace_count, activities_to_replace_with_count
    )

    # 3: compute for all logs all activity distance matrices
    # activity_distance_functions we want to evaluate
    activity_distance_functions = ["Bose 2009 Substitution Scores"]

    activity_distance_matrix_dict = get_activity_distance_matrix_dict(
        activity_distance_functions, logs_with_replaced_activities_dict
    )

    # 4: evaluation of all activity distance matrices
    knn_dict = get_knn_dict(activity_distance_matrix_dict, activities_to_replace_with_count)
    precision_at_k_dict = get_precision_at_k(knn_dict, activity_distance_functions)

    print(precision_at_k_dict)

    precision = precision_at_k_dict["Bose 2009 Substitution Scores"]

    # Clean up to save memory
    del logs_with_replaced_activities_dict
    gc.collect()

    return different_activities_to_replace_count, activities_to_replace_with_count, precision


if __name__ == '__main__':
    log = xes_importer.apply(ROOT_DIR + '/event_logs/repairExample.xes')
    log_control_flow_perspective = get_log_control_flow_perspective(log)
    alphabet = get_alphabet(log_control_flow_perspective)

    combinations = [
        (
        different_activities_to_replace_count, activities_to_replace_with_count, log_control_flow_perspective, alphabet)
        for different_activities_to_replace_count in range(1, len(alphabet))
        for activities_to_replace_with_count in range(2, 10)
    ]

    with Pool() as pool:
        results = pool.map(process_combination, combinations)

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=['r', 'w', 'precision@'])
    result = df.pivot(index='w', columns='r', values='precision@')

    # Plotting
    f, ax = plt.subplots(figsize=(11, 9))
    rc('font', **{'family': 'serif', 'size': 20})
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, annot=True, linewidth=.5)
    ax.invert_yaxis()

    plt.savefig("Histo.pdf", format="pdf", transparent=True)
    plt.show()
