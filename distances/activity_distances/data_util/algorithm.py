import os
from typing import List
from definitions import ROOT_DIR
import csv
from datetime import datetime, timedelta

#Given Pm4py Event Log, return List of Lists of Activities (List of Traces)
def give_log_padding(log, ngram_size):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    middle_index = ngram_size // 2 + 1
    if middle_index % 2 == 1:
        padding_left = ["."]*(ngram_size-middle_index)
        padding_right = padding_left
    else:
        padding_left = ["."]*(ngram_size-middle_index + 1)
        padding_right = ["."]*(ngram_size-middle_index)
    for trace in log:
        #adjust for different ngram size
        log_list[i].extend(padding_left)
        log_list[i].extend(trace)
        log_list[i].extend(padding_right)
        i += 1
    return log_list


def transform_control_flow_lists_to_csv(control_flow_lists):
    # Convert event log to CSV format
    start_time = datetime(1970, 1, 1)
    time_delta = timedelta(hours=1)

    events = []
    for case_id, activities in enumerate(activity_sequences):
        for event_index, activity in enumerate(activities):
            timestamp = start_time + event_index * time_delta
            events.append([case_id, activity, timestamp.isoformat() + "+00:00"])

    process_id = os.getpid()
    # Output CSV file name
    output_file = f"event_log_{process_id}.csv"


    # Write to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["CaseID", "Activity", "Timestamp"])
        writer.writerows(events)

    return output_file



