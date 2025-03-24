import os
import pickle
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import pandas as pd

EVENT_LOGS_DIR = os.path.join(ROOT_DIR , "evaluation", "evaluation_of_activity_distances", "event_logs", "intrinsic_evaluation", "results")

all_logs = ['BPIC12',
            'BPIC12_A',
            'BPIC12_Complete',
            'BPIC12_O',
            'BPIC12_W',
            'BPIC12_W_Complete',
            'BPIC13_closed_problems',
            'BPIC13_incidents',
            'BPIC13_open_problems',
            'BPIC15_1',
            'BPIC15_2',
            'BPIC15_3',
            'BPIC15_4',
            'BPIC15_5',
            'BPIC17',
            'BPIC18',
            'BPIC19',
            'BPIC20_DomesticDeclarations',
            'BPIC20_InternationalDeclarations',
            'BPIC20_PermitLog',
            'BPIC20_PrepaidTravelCost',
            'BPIC20_RequestForPayment',
            'CCC19',
            'Env Permit',
            'Helpdesk',
            'Hospital Billing',
            'RTFM',
            'Sepsis']

log_statistics = {}

"""
window_size_list = [3,5,9]
r = 7
w = 10
sampling_size = 10
for log_name in all_logs:
    df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", log_name)
os.makedirs(df_avg_dir, exist_ok=True)
file_name = f"dfavg_r{r}_w{w}_samplesize_{sampling_size}.pkl"
file_path = os.path.join(df_avg_dir, file_name)

if os.path.isfile(file_path):
    logs_with_replaced_activities_dict = pickle.load(open(file_path, "rb"))
    print("a")
"""

df_list = []

# Loop through each log directory
for log_name in all_logs:
    df_avg_dir = os.path.join(ROOT_DIR, "results", "activity_distances", "intrinsic_df_avg", log_name)

    if os.path.exists(df_avg_dir):  # Check if directory exists
        # Find the pickle file
        pickle_files = [f for f in os.listdir(df_avg_dir) if f.endswith(".pkl")]

        if pickle_files:  # If a pickle file exists
            pickle_path = os.path.join(df_avg_dir, pickle_files[0])  # Assuming only one pickle file per directory
            df = pd.read_pickle(pickle_path)  # Load DataFrame
            df_list.append(df)  # Append to the list

# Merge all DataFrames
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)

    # Compute the average per Distance Function
    avg_df = combined_df.groupby('Distance Function')[['diameter', 'precision@w-1', 'precision@1', 'triplet']].mean()

    # Plot bar charts for each metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['diameter', 'precision@w-1', 'precision@1', 'triplet']
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        avg_df[metric].plot(kind='bar', ax=axes[i], title=f'Average {metric} by Distance Function', color='skyblue')
        axes[i].set_ylabel(metric)

    plt.tight_layout()
    plt.show()
else:
    print("No valid dataframes found in the specified directories.")
