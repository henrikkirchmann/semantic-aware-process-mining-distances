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
from distances.activity_distances.chiorrini_2023_embedding_process_structure.embedding_process_structure import (
    get_embedding_process_structure_distance_matrix
)


def evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions):
    results = []

    for log_name in log_list:
        # Import the event log
        log = xes_importer.apply(ROOT_DIR + '/event_logs/' + log_name + '.xes')
        log_control_flow_perspective = get_log_control_flow_perspective(log)
        alphabet = get_alphabet(log_control_flow_perspective)

        for activity_distance_function in activity_distance_functions:
            runtimes = []
            for _ in range(number_of_repetitions):

                # Evaluate the activity distance function
                if activity_distance_function == "Bose 2009 Substitution Scores":
                    start_time = time.time()
                    get_substitution_and_insertion_scores(
                        log_control_flow_perspective,
                        alphabet,
                        3
                    )
                    runtimes.append(time.time() - start_time)
                elif activity_distance_function.startswith("De Koninck 2018 act2vec"):
                    sg = 0 if "CBOW" in activity_distance_function else 1
                    start_time = time.time()
                    get_act2vec_distance_matrix(log_control_flow_perspective, alphabet, sg)
                    runtimes.append(time.time() - start_time)
                elif activity_distance_function == "Unit Distance":
                    start_time = time.time()
                    get_unit_cost_activity_distance_matrix(log_control_flow_perspective, alphabet)
                    runtimes.append(time.time() - start_time)
                elif activity_distance_function == "Chiorrini 2023 Embedding Process Structure":
                    # to make the authors implmenetation work we have to some I/O operations, with pnml files,
                    # thus we measure the time without them, and implemented the time measure inside the called function
                    runtimes.append(get_embedding_process_structure_distance_matrix(log_control_flow_perspective, alphabet, True))


            # Calculate the average runtime
            avg_runtime = sum(runtimes) / len(runtimes)
            results.append({
                "activity_function_name": activity_distance_function,
                "log": log_name,
                "average_duration": avg_runtime
            })

    return results


if __name__ == '__main__':
    # Define the activity distance functions to evaluate
    activity_distance_functions = [
         "Bose 2009 Substitution Scores",
         "De Koninck 2018 act2vec CBOW",
        "Chiorrini 2023 Embedding Process Structure",
         "Unit Distance",
         "De Koninck 2018 act2vec skip-gram"
    ]

    # Define the logs to evaluate
    log_list = [
        "repairExample",
         "Sepsis",
        # "bpic_2015",
        # "Road Traffic Fine Management Process",
    ]

    number_of_repetitions = 10

    # Evaluate runtimes
    print(f"Evaluating runtimes with {number_of_repetitions} repetitions...")
    runtime_results = evaluate_runtime(activity_distance_functions, log_list, number_of_repetitions)

    # Save results to CSV
    csv_filename = f"runtime_results_{number_of_repetitions}_repetitions.csv"
    df = pd.DataFrame(runtime_results)
    df.to_csv(ROOT_DIR + '/results/' + csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
