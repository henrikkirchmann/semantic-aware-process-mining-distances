from pm4py.statistics.variants.log import get as variants_module
#from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer

from distances.activity_distances.bose_2009_context_aware_trace_clustering.algorithm import get_substitution_and_insertion_scores

log = xes_importer.apply('//repairExample.xes')



#Trace Distance with Levenshtein and Bose 2009

substitution_scores = get_substitution_and_insertion_scores(log, 3)

language = variants_module.get_language(log)

print("a")
#emd = emd_evaluator.apply(language, language)