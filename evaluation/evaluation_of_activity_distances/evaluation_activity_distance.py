from pm4py.statistics.variants.log import get as variants_module
#from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer
from evaluation.data_util.algorithm import get_log_control_flow_perspective, get_alphabet

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import get_substitution_and_insertion_scores

from definitions import ROOT_DIR

log = xes_importer.apply(ROOT_DIR + '/repairExample.xes')

log_control_flow_perspective = get_log_control_flow_perspective(log)#transform log to a list of lists of activity labels
alphabet = get_alphabet(log_control_flow_perspective)


#count_different_activities_to_replace =
#1: get activities we want to replace in each run
alphabet


#Trace Distance with Levenshtein and Bose 2009
substitution_scores = get_substitution_and_insertion_scores(log_control_flow_perspective, alphabet,5)





language = variants_module.get_language(log)

print("a")
#emd = emd_evaluator.apply(language, language)