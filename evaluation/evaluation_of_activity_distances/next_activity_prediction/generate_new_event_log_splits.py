import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from pm4py.objects.log.obj import EventLog

# Replace your current split_log function with this one
from pm4py.objects.log.obj import EventLog

def split_log(log, train_frac=0.64, val_frac=0.16):
    total_length = len(log)
    train_end = int(total_length * train_frac)
    val_end = train_end + int(total_length * val_frac)

    train_log = EventLog(log[:train_end])
    val_log = EventLog(log[train_end:val_end])
    test_log = EventLog(log[val_end:])
    train_val_log = EventLog(log[:val_end])

    # Update properties individually
    for split in [train_log, val_log, test_log, train_val_log]:
        split.extensions.update(log.extensions)
        split.classifiers.update(log.classifiers)
        split.attributes.update(log.attributes)

    return train_log, val_log, test_log, train_val_log


import os

# Get the absolute path of the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define directories relative to the script's location
raw_dir = os.path.join(script_dir, "raw_datasets")
split_dir = os.path.join(script_dir, "split_datasets")
os.makedirs(split_dir, exist_ok=True)

# Process each XES file in raw_datasets
for file_name in os.listdir(raw_dir):
    if file_name.endswith(".xes.gz"):
        file_path = os.path.join(raw_dir, file_name)

        # Import XES file
        log = xes_importer.apply(file_path)

        # Perform splits
        train_log, val_log, test_log, train_val_log = split_log(log)

        base_name = file_name.replace(".xes.gz", "")

        # Export logs as .xes.gz
        xes_exporter.apply(train_log, os.path.join(split_dir, f"train_{base_name}.xes.gz"), parameters={"compress": True})
        xes_exporter.apply(val_log, os.path.join(split_dir, f"val_{base_name}.xes.gz"), parameters={"compress": True})
        xes_exporter.apply(test_log, os.path.join(split_dir, f"test_{base_name}.xes.gz"), parameters={"compress": True})
        xes_exporter.apply(train_val_log, os.path.join(split_dir, f"train_val_{base_name}.xes.gz"), parameters={"compress": True})

print("Splitting and saving completed.")
