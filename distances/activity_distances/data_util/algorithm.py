from typing import List

def get_all_activities_from_list_of_traces_that_have_padding(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        #adjust for different ngram size
        for activity in trace[1:-1]:
            unique_activities.add(activity)
    return list(unique_activities)

#Given Pm4py Event Log, return List of Lists of Activities (List of Traces)
def transform_log_to_trace_string_list_with_padding(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        #adjust for different ngram size
        log_list[i].append(".")
        for event in trace._list:
            log_list[i].append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            #log_list[i].append(event._dict.get('concept:name'))
        log_list[i].append(".")
        i += 1
    return log_list


