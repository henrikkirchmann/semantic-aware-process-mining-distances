import os
import pandas as pd
import matplotlib.pyplot as plt
from definitions import ROOT_DIR

# Define the main path and ignore folders
base_path = ROOT_DIR + '/results/activity_distances/intrinsic'
ignore_folders = {'nn', 'precision_at_k'}
distance_functions = ["Bose 2009 Substitution Scores", "De Koninck 2018 act2vec CBOW"]

# Data structures to store precision averages for each distance function and each folder
precision_1_results = {func: {} for func in distance_functions}
precision_w1_results = {func: {} for func in distance_functions}

# Loop through each folder in the base path
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    # Skip if it's an ignored folder or not a directory
    if folder_name in ignore_folders or not os.path.isdir(folder_path):
        continue

    # Initialize dictionaries to store cumulative precision@1 and precision@w-1 values for each w
    folder_precision_1 = {func: {} for func in distance_functions}
    folder_precision_w1 = {func: {} for func in distance_functions}
    count_per_w = {func: {} for func in distance_functions}  # Track counts per w for averaging

    # Process each file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Identify the distance function from the file name
        for func in distance_functions:
            if func in file_name:  # Match the file with the function based on naming convention
                # Load CSV file into DataFrame
                df = pd.read_csv(file_path)

                average_precisions = df.groupby('w')[['precision@w-1', 'precision@1']].mean().reset_index()
                average_precisions['log'] = folder_name

                # Accumulate precision values for averaging
                for _, row in df.iterrows():
                    w = row['w']

                    # Initialize dictionaries if encountering w for the first time
                    if w not in folder_precision_1[func]:
                        folder_precision_1[func][w] = 0
                        folder_precision_w1[func][w] = 0
                        count_per_w[func][w] = 0

                    # Accumulate precision values and increment count
                    folder_precision_1[func][w] += row['precision@1']
                    folder_precision_w1[func][w] += row['precision@w-1']
                    count_per_w[func][w] += 1
                break  # Exit the loop after finding the matching function for this file

    # Calculate average precision for each distance function in this folder
    for func in distance_functions:
        precision_1_results[func][folder_name] = []
        precision_w1_results[func][folder_name] = []

        for w in folder_precision_1[func]:
            avg_precision_1 = folder_precision_1[func][w] / count_per_w[func][w]
            avg_precision_w1 = folder_precision_w1[func][w] / count_per_w[func][w]

            precision_1_results[func][folder_name].append((w, avg_precision_1))
            precision_w1_results[func][folder_name].append((w, avg_precision_w1))

# Plotting the results
for func in distance_functions:
    # Plot average precision@1 values
    plt.figure(figsize=(10, 6))
    for folder_name, values in precision_1_results[func].items():
        w_values, precision_1_values = zip(*sorted(values))
        plt.plot(w_values, precision_1_values, marker='o', label=folder_name)
    plt.grid(True)
    plt.title(f'Average Precision@1 for {func}')
    plt.xlabel('w')
    plt.ylabel('Average Precision@1')
    plt.legend(title='Folders')
    plt.show()

    # Plot average precision@w-1 values
    plt.figure(figsize=(10, 6))
    for folder_name, values in precision_w1_results[func].items():
        w_values, precision_w1_values = zip(*sorted(values))
        plt.plot(w_values, precision_w1_values, marker='o', label=folder_name)
    plt.grid(True)
    plt.title(f'Average Precision@w-1 for {func}')
    plt.xlabel('w')
    plt.ylabel('Average Precision@w-1')
    plt.legend(title='Folders')
    plt.show()
