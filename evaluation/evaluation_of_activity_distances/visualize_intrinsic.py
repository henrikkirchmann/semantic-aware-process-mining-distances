from pathlib import Path

import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt, rc

from definitions import ROOT_DIR

def visualization_intrinsic_evaluation_from_csv(log_name):
    # Specify the file name that was used earlier
    for file in os.listdir(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name):
       
        # Step 1: Remove the ".csv" extension
        input_str = file.replace(".csv", "")

        # Step 2: Split the string by underscores
        parts = input_str.split("_")

        # Step 3: Assign the parts to the variables
        log_name = parts[0]
        activity_distance_function = parts[2]
        r = int(parts[3][1:])  # Extract the number after 'r'
        w = int(parts[4][1:])  # Extract the number after 'w'
        sampling_size = int(parts[6])  # Extract the number after 'samplesize'
        print(f"r: {r}, w: {w}, sampling_size: {sampling_size}")

        # Load the DataFrame from the CSV file
        df = pd.read_csv(ROOT_DIR + '/results/activity_distances/intrinsic/' + log_name + "/" + file)

        # heat map precision@w-1
        result = df.pivot(index='w', columns='r', values='precision@w-1')
        average_value = result.values.mean()
        print("Average precision@w-1 is: " + str(average_value) + " " + activity_distance_function)

        # Plotting
        rc('font', **{'family': 'serif', 'size': 20 * 3.5})
        f, ax = plt.subplots(figsize=(17 + 17 * int(r / 17), 20))
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, linewidth=.5)
        ax.invert_yaxis()
        ax.set_title("precision@w-1 for " + log_name + " with max sampling size " + str(
            sampling_size) + "\n" + activity_distance_function, pad=20)
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            ROOT_DIR + "/results/activity_distances/intrinsic/precision_at_k/" + "pre_" + activity_distance_function + "_" + log_name + "_r:" + str(
                r) + "_w:" + str(w) + "_sampling:" + str(sampling_size) + ".pdf", format="pdf", transparent=True)
        plt.show()

        # heat map precision@1
        result = df.pivot(index='w', columns='r', values='precision@1')
        average_value = result.values.mean()
        print("Average Nearest Neighbor is: " + str(average_value) + " " + activity_distance_function)
        # Plotting
        rc('font', **{'family': 'serif', 'size': 20 * 3})
        f, ax = plt.subplots(figsize=(17 + 17 * int(r / 17), 20))
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(result, cmap=cmap, vmin=0, vmax=1, linewidth=.5)
        ax.invert_yaxis()
        ax.set_title("Nearest Neighbor for " + log_name + " with max sampling size " + str(
            sampling_size) + "\n" + activity_distance_function, pad=20)
        Path(ROOT_DIR + "/results/activity_distances/intrinsic/nn").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            ROOT_DIR + "/results/activity_distances/intrinsic/nn/" + "nn" + activity_distance_function + "_" + log_name + "_r:" + str(
                r) + "_w:" + str(w) + "_sampling:" + str(sampling_size) + ".pdf", format="pdf", transparent=True)
        plt.show()

if __name__ == '__main__':
    #log_name = "BPI Challenge 2018"
    #log_name = "Road Traffic Fine Management Process"
    log_name = "Sepsis"
    print(log_name)
    visualization_intrinsic_evaluation_from_csv(log_name)

