import time
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from definitions import ROOT_DIR
from evaluation.data_util.util_activity_distances import (
    get_alphabet,
    get_unit_cost_activity_distance_matrix
)
from evaluation.data_util.util_activity_distances_intrinsic import get_log_control_flow_perspective
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import (
    get_substitution_and_insertion_scores
)
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import (
    get_embedding_process_structure_distance_matrix
)
import gc
import math
import os
import shutil
import sys
from collections import defaultdict
from typing import List
import re
import psutil
import numpy as np
from definitions import ROOT_DIR
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import \
    get_activity_activity_co_occurence_matrix
from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import \
    get_substitution_and_insertion_scores
from distances.activity_distances.chiorrini_2022_embedding_process_structure.embedding_process_structure import \
    get_embedding_process_structure_distance_matrix
from distances.activity_distances.de_koninck_2018_act2vec.algorithm import get_act2vec_distance_matrix
from distances.activity_distances.de_koninck_2018_act2vec.our_hyperparas import get_act2vec_distance_matrix_our
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.embeddings_generator.main_new import \
    get_context_based_distance_matrix
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import \
    get_activity_context_frequency_matrix
from distances.activity_distances.pmi.pmi import get_activity_context_frequency_matrix_pmi, \
    get_activity_activity_frequency_matrix_pmi
from evaluation.data_util.util_activity_distances import extract_window_size


def evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions):
    results = []

    for log_name in log_list:
        # Import the event log
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes.gz')
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        alphabet = get_alphabet(log_control_flow_perspective)

        for activity_distance_function in activity_distance_functions:
            runtimes = []
            for _ in range(number_of_repetitions):
                window_size = extract_window_size(activity_distance_function)

                if activity_distance_function.startswith("Bose 2009 Substitution Scores"):
                    start_time = time.time()
                    activity_distance_matrix, embedding = get_substitution_and_insertion_scores(
                        log_control_flow_perspective,
                        alphabet, window_size)
                    runtimes.append(time.time() - start_time)
                elif activity_distance_function.startswith("De Koninck 2018 act2vec"):
                    if "CBOW" in activity_distance_function:
                        sg = 0
                    else:
                        sg = 1
                    start_time = time.time()
                    act2vec_distance_matrix, embedding = get_act2vec_distance_matrix(
                        log_control_flow_perspective,
                        alphabet, sg, window_size)
                    runtimes.append(time.time() - start_time)

                elif activity_distance_function.startswith("Unit Distance"):
                    start_time = time.time()

                    unit_distance_matrix, emb = get_unit_cost_activity_distance_matrix(
                        log_control_flow_perspective,
                        alphabet)
                    runtimes.append(time.time() - start_time)

                elif "Chiorrini 2022 Embedding Process Structure" in activity_distance_function:
                    # to make the authors implmenetation work we have to some I/O operations, with pnml files,
                    # thus we measure the time without them, and implemented the time measure inside the called function
                    # 1 for time measurement with pm discovery (inductive miner)
                    # 2 for time measurement without pm discovery
                    if "Discovery" in activity_distance_function:
                        runtimes.append(
                            get_embedding_process_structure_distance_matrix(log_control_flow_perspective, alphabet, 1))
                        print("w discovery")
                    else:
                        runtimes.append(
                            get_embedding_process_structure_distance_matrix(log_control_flow_perspective, alphabet, 2))
                        print("wo discovery")


                elif "Gamallo Fernandez 2023 Context Based" in activity_distance_function:
                    runtimes.append(get_context_based_distance_matrix(
                        log_control_flow_perspective, window_size, take_time=True))
                elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence Bag Of Words"):
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, True)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         0)
                        runtimes.append(time.time() - start_time)

                    else:
                        runtimes.append(time.time() - start_time)


                elif activity_distance_function.startswith("Activity-Activitiy Co Occurrence N-Gram"):
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, activity_index = get_activity_activity_co_occurence_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, False)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         1)
                        runtimes.append(time.time() - start_time)
                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_activity_frequency_matrix_pmi(embedding,
                                                                                                         activity_freq_dict,
                                                                                                         activity_index,
                                                                                                         0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)

                elif activity_distance_function.startswith("Activity-Context Bag Of Words"):
                    start_time = time.time()
                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 2)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)



                elif "Activity-Context as Bag of Words as N-Grams" in activity_distance_function:
                    start_time = time.time()

                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 1)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)
                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)
                elif "Activity-Context N-Grams" in activity_distance_function:
                    start_time = time.time()
                    activity_distance_matrix, embedding, activity_freq_dict, context_freq_dict, context_index = get_activity_context_frequency_matrix(
                        log_control_flow_perspective,
                        alphabet, window_size, 0)
                    if "PPMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        1)
                        runtimes.append(time.time() - start_time)

                    elif "PMI" in activity_distance_function:
                        activity_distance_matrix, embedding = get_activity_context_frequency_matrix_pmi(embedding,
                                                                                                        activity_freq_dict,
                                                                                                        context_freq_dict,
                                                                                                        context_index,
                                                                                                        0)
                        runtimes.append(time.time() - start_time)
                    else:
                        runtimes.append(time.time() - start_time)
                else:
                    raise ValueError("Unknown encoding method: " + activity_distance_function)

            # Calculate the average runtime
            avg_runtime = sum(runtimes) / len(runtimes)
            print({
                "activity_function_name": activity_distance_function,
                "log": log_name,
                "average_duration": avg_runtime
            })
            results.append({
                "activity_function_name": activity_distance_function,
                "log": log_name,
                "average_duration": avg_runtime
            })

    return results


if __name__ == '__main__':
    # Define the activity distance functions to evaluate
    activity_distance_functions = list()
    activity_distance_functions.append("Unit Distance")
    activity_distance_functions.append("Bose 2009 Substitution Scores")
    activity_distance_functions.append("De Koninck 2018 act2vec CBOW")
    activity_distance_functions.append("Chiorrini 2022 Embedding Process Structure")
    activity_distance_functions.append("Chiorrini 2022 Embedding Process Structure Discovery")

    activity_distance_functions.append("De Koninck 2018 act2vec skip-gram")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence Bag Of Words PPMI")
    activity_distance_functions.append("Activity-Activitiy Co Occurrence N-Gram PPMI")
    activity_distance_functions.append("Activity-Context Bag Of Words")
    activity_distance_functions.append("Activity-Context N-Grams")
    activity_distance_functions.append("Activity-Context Bag Of Words PMI")
    activity_distance_functions.append("Activity-Context N-Grams PMI")
    activity_distance_functions.append("Activity-Context Bag Of Words PPMI")
    activity_distance_functions.append("Activity-Context N-Grams PPMI")

    from evaluation.data_util.util_activity_distances_intrinsic import add_window_size_evaluation

    window_size_list = [3, 5, 9]

    activity_distance_functions = add_window_size_evaluation(activity_distance_functions, window_size_list)

    activity_distance_functions.append("Gamallo Fernandez 2023 Context Based w_3")

    # Define the logs to evaluate
    log_list = [
        'Sepsis'
    ]


    number_of_repetitions = 5

    # Evaluate runtimes
    print(f"Evaluating runtimes with {number_of_repetitions} repetitions...")
    runtime_results = evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions)

    # Save results to CSV
    csv_filename = f"runtime_results_{number_of_repetitions}_repetitions_gamallo_19.csv"
    df = pd.DataFrame(runtime_results)
    df.to_csv(ROOT_DIR + '/results/' + csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
