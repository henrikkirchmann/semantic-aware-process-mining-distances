from typing import List, Tuple, Dict

def get_log_control_flow_perspective(log):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    for trace in log:
        for event in trace._list:
            #log_list[i].append(event._dict.get('concept:name')+"-"+event._dict.get('lifecycle:transition'))
            log_list[i].append(event._dict.get('concept:name'))
        i += 1
    return log_list

def get_alphabet(log: List[List[str]]) -> List[str]:
    unique_activities = set()
    for trace in log:
        # adjust for different ngram size
        for activity in trace:
            unique_activities.add(activity)
    return list(unique_activities)


