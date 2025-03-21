import os
import time

import numpy as np
import pm4py
import pm4py.objects.process_tree.utils.generic as generic
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY

from distances.activity_distances.chiorrini_2022_embedding_process_structure.model_feature import optionality, p_length
from distances.activity_distances.chiorrini_2022_embedding_process_structure.new_parallelism_and_pathlength import \
    newparallelism, new_parallelism_pathlength, new_pathlength, new_pathlength_in_place
from distances.activity_distances.chiorrini_2022_embedding_process_structure.tree_feature import make_visible, \
    feature_map
from distances.activity_distances.data_util.algorithm import get_cosine_distance_dict
from tabulate import tabulate


def get_embedding_process_structure_distance_matrix(log, alphabet, take_time):
    event_log = EventLog()
    # Transform the list of traces into an EventLog object
    for trace_id, trace in enumerate(log):
        pm4py_trace = Trace()
        for event_id, activity in enumerate(trace):
            # Create an event with attributes
            event = Event({
                DEFAULT_NAME_KEY: activity,  # 'concept:name' for activity name
                "trace_id": trace_id,  # Custom trace attribute
                "event_index": event_id  # Index of the event in the trace
            })
            pm4py_trace.append(event)  # Add event to the trace
        event_log.append(pm4py_trace)  # Add trace to the event log

    process_id = os.getpid()

    start_time = time.time()

    # Discover the workflow net
    net_or, im, fm = pm4py.discover_petri_net_inductive(event_log)
    net_original_file = f"temp_petri_net_{process_id}.pnml"
    pm4py.write_pnml(net_or, im, fm, net_original_file)
    net_modified_file = net_original_file.replace(".pnml", "_visible.pnml")

    with open(net_original_file) as infile:
        with open(net_modified_file, 'w') as outfile:
            outfile.write(make_visible(infile.read()))

    net, initial_marking, final_marking = pm4py.read_pnml(net_modified_file)
    net_or, im, fm = pm4py.read_pnml(net_original_file)  # IMPORTANT: be sure to use the original net

    # Clean up the temporary file
    os.remove(net_original_file)
    os.remove(net_modified_file)

    tree = wf_net_converter.apply(net, initial_marking, final_marking)
    tree_2 = generic.fold(tree)

    op = tree_2._get_operator()
    # curr_features = (1, 1, 0, 0)
    ris = feature_map(tree_2)

    out = {}
    for name in ris.keys():
        if name.label is not None:
            l = name.label
            out[l] = ris[name]

    start = time.time()
    # Feature indices
    id_par = 0
    id_opt = 1
    id_sloop = 2
    id_lloop = 3

    #start = time.perf_counter()
    #path_l = p_length(out, net_or, im)
    #end = time.perf_counter()
    #print(f"path_l computation time: {end - start:.6f} seconds")

    #start = time.perf_counter()
    path_l = new_pathlength_in_place(tree_2)
    #end = time.perf_counter()
    #print(f"path_in_place computation time: {end - start:.6f} seconds")

    #data = [[key, path_l[key], path_in_place[key], path_in_place[key] - path_l[key]] for key in path_l]
    #print(tabulate(data, headers=["Key", "Path_L", "Path_In_Place", "Difference"], tablefmt="grid"))


    # Measure optionality computation time
    opt = optionality(out, id_opt)

    # Measure new parallelism computations
    new_parallelism_pathlength_dict = new_parallelism_pathlength(tree_2)
    newparallelism_dict = newparallelism(tree_2)

    # Features computation
    features = {}
    for elem in out:
        if 'tau' in elem or "Inv" in elem:
            continue

        m = []
        m.append(path_l.get(elem, 0))
        m.append(opt.get(elem, 1))
        m.append(new_parallelism_pathlength_dict.get(elem, 0))
        m.append(newparallelism_dict.get(elem, 0))
        m.append(out[elem][id_sloop])
        m.append(out[elem][id_lloop])

        features[elem] = np.array(m)

    # Compute distances
    distances = get_cosine_distance_dict(features)

    if take_time is False:
        return distances, features
    else:
        return time.time() - start_time


def cosine_distance(array1, array2):
    # Compute the dot product and magnitudes
    dot_product = np.dot(array1, array2)
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)

    # Compute cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:  # Handle zero vectors
        return 1.0  # Maximum cosine distance for orthogonal vectors
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    # Compute cosine distance
    return 1 - cosine_similarity
