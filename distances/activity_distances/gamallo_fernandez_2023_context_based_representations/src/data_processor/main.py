"""
DATA PROCESSOR:
Read a XES file, convert it to a CSV file and make the splits.
It can create holdout or cross-validation partitions.
"""

from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.args_reader import read_input_args
from distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.logger import DataProcessorLogger
import distances.activity_distances.gamallo_fernandez_2023_context_based_representations.src.data_processor.utils as utils
import sys
import os


def write_csvs(xes_file_name, output_file):
    #xes_file_path = os.path.abspath(os.path.join("data/raw_datasets", xes_file_name))

    sys.argv = [
        "script.py",  # Fake script name (needed for argparse)
        "--dataset", output_file,  # Single dataset mode
        "--crossvalidation",  # Enables cross-validation
        "--activity",  # Extract ActivityID column
        "--timestamp",  # Extract Timestamp column
        "--stats",  # Print dataset statistics
        "--print_console_file"  # Print output to both console and file
    ]

    args = read_input_args()

    logger = DataProcessorLogger(args.print_mode)

    # Get the list of datasets to be processed
    dataset_list = utils.get_datasets_list(args.xes_path, args.batch, logger)

    # Select the specified columns
    output_columns = utils.select_output_columns(args.activity, args.timestamp,
                                                 args.resource)

    # Loop iterating over each dataset
    for xes_dataset in dataset_list:
        csv_dataset, config_csv_path, attr_dict  = utils.convert_xes_to_csv(xes_dataset, output_columns)

        # Make the partitions
        if args.crossvalidation:
            utils.make_crossvalidation(csv_dataset)
        else:
            utils.make_holdout(csv_dataset)

        # Calculate the statistics of the dataset
        #if args.stats:
        #    stats = utils.calculate_stats(csv_dataset, logger)
    return csv_dataset, config_csv_path, attr_dict

