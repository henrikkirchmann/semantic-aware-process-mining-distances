from typing import List


#Given Pm4py Event Log, return List of Lists of Activities (List of Traces)
def give_log_padding(log, ngram_size):
    log_list = list()
    for trace in log:
        log_list.append(list())
    i = 0
    middle_index = ngram_size // 2 + 1
    if middle_index % 2 == 1:
        padding_left = ["."]*(ngram_size-middle_index)
        padding_right = padding_left
    else:
        padding_left = ["."]*(ngram_size-middle_index + 1)
        padding_right = ["."]*(ngram_size-middle_index)
    for trace in log:
        #adjust for different ngram size
        log_list[i].extend(padding_left)
        log_list[i].extend(trace)
        log_list[i].extend(padding_right)
        i += 1
    return log_list


