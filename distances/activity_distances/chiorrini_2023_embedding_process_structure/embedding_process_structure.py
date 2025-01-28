import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_import
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
import pm4py.objects.process_tree.utils.generic as generic
from distances.activity_distances.chiorrini_2023_embedding_process_structure.configuration import import_path, file_print
from distances.activity_distances.chiorrini_2023_embedding_process_structure.model_feature import p_length, optionality, parallelism
from distances.activity_distances.chiorrini_2023_embedding_process_structure.tree_feature import make_visible, feature_map
import time
import numpy as np
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from distances.activity_distances.chiorrini_2023_embedding_process_structure.new_parallelism import newparallelism, new_parallelism_pathlength
from io import StringIO
from pm4py.objects.petri_net.exporter.exporter import apply as export_pnml
from pm4py.objects.petri_net.importer.importer import apply as import_pnml



def get_embedding_process_structure_distance_matrix(log, alphabet):

    logname = "logname.xes"
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

    # Discover the workflow net
    net_or, im, fm = pm4py.discover_petri_net_inductive(event_log)

    # Serialize the Petri net to a string (PNML format)
    pnml_buffer = StringIO()
    export_pnml(net_or, im, fm, pnml_buffer)
    pnml_string = pnml_buffer.getvalue()
    pnml_buffer.close()

    # Apply modifications to the PNML string
    pnml_modified_string = make_visible(pnml_string)

    # Deserialize the modified PNML string back into a Petri net
    pnml_modified_buffer = StringIO(pnml_modified_string)
    net, initial_marking, final_marking = import_pnml(pnml_modified_buffer)
    pnml_modified_buffer.close()


    tree = wf_net_converter.apply(net, initial_marking, final_marking)
    #pm4py.view_process_tree(tree)
    tree_2 = generic.fold(tree)
    #pm4py.view_process_tree(tree_2)

    op = tree_2._get_operator()
    #curr_features = (1, 1, 0, 0)
    ris = feature_map(tree_2)

    out = {}
    for name in ris.keys():
        if name.label is not None:
            l = name.label
            out[l] = ris[name]


    # feature indices
    id_par = 0
    id_opt = 1
    id_sloop = 2
    id_lloop = 3

    t1 = time.time()
    path_l = p_length(out, net_or, im)
    t2 = time.time()
    t_dist = round(t2 - t1, 2)
    #print("Path Length elaboration time: ", t_dist)

    opt = optionality(out, id_opt)
    t3 = time.time()
    t_opt = round(t3 - t2, 2)
    #print("Optionality, elaboration time: ", t_opt)

    new_parallelism_pathlength_dict = new_parallelism_pathlength(tree_2)

    newparallelism_dict = newparallelism(tree_2)

    #paral_mod = parallelism(tree_2, net, out, id_par)
    t4 = time.time()
    t_par = round(t4 - t3, 2)
    #print("Parallelism, elaboration time: ", t_par)

    features = {}
    #print()
    #print("Activities features: ")
    #"name activity;path length;optionality;par path length;parallelism;strectly loopable;long loopable;"
    for elem in out:
        if 'tau' in elem or "Inv" in elem: #or 'END' in elem or 'START' in elem:
            continue
        #print(elem)
        # ASUBMITTED
        #if elem == "ASUBMITTED" or elem == "APARTLYSUBMITTED":
        #    print("a")
        m = []
        if elem in path_l:
            m.append(path_l[elem])
        else:
            m.append(0)
        if elem in opt:
            m.append(opt[elem])
        else:
            m.append(1)
        if elem in new_parallelism_pathlength_dict:
            m += [new_parallelism_pathlength_dict[elem]]
        else:
            m += [0]
        if elem in newparallelism_dict:
            m += [newparallelism_dict[elem]]
        else:
            m += [0]
        m.append(out[elem][id_sloop])
        m.append(out[elem][id_lloop])

        np_m = np.array(m)
        features[elem] = np_m

    #print(features)
    distances = {}
    for activity1 in alphabet:
        for activity2 in alphabet:
            distance = cosine_distance(features[activity1], features[activity2])
            distances[(activity1, activity2)] = distance
    return distances


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


