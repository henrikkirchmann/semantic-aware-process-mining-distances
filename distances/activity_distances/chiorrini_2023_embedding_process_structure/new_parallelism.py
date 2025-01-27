from pm4py.objects.process_tree import obj as pt_opt
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.process_tree import obj as pt_opt
from pm4py.objects.process_tree import state as pt_st
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.utils import generic as pt_util
from tensorflow.python.keras.saving.saved_model_experimental import sequential
from itertools import product, permutations

'''
Re-Implementation of Parallelism & Parallelism Path Length Feature, 
because original implementation works only with their given Petrinets.
'''

class GenerationTree(ProcessTree):
    # extend the parent class to replace the __eq__ and __hash__ method

    def __init__(self, tree):
        i = 0
        self.parallel_ancestor_node_list = []
        self.number_of_parallel_branches = None
        self.sub_tree_id_of_parallel_ancestor_node_list = []

        while i < len(tree.children):
            tree.children[i] = GenerationTree(tree.children[i])
            tree.children[i].parent = self
            i = i + 1
        ProcessTree.__init__(self, operator=tree.operator, parent=tree.parent, children=tree.children, label=tree.label)

    def __eq__(self, other):
        # method that is different from default one (different taus must give different ID in log generation!!!!)
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


def newparallelism(process_tree):
    tree = GenerationTree(process_tree)
    #depth first search
    to_visit = [tree]
    while len(to_visit) > 0:
        n = to_visit.pop(0)
        if n.operator is pt_opt.Operator.PARALLEL:
            if len(n.parallel_ancestor_node_list) == 0:
                i = 0
                for child in n.children:
                    propagateParallelism([n], child, [i], len(n.children))
                    i = i + 1
            else:
                parallel_ancestor_node_list = n.parallel_ancestor_node_list + [n]
                i = 0
                for child in n.children:
                    sub_tree_id_of_parallel_ancestor_node_list = n.sub_tree_id_of_parallel_ancestor_node_list + [i]
                    propagateParallelism(parallel_ancestor_node_list, child, sub_tree_id_of_parallel_ancestor_node_list, len(n.children))
                    i = i + 1

                for i in range(0,len(n.sub_tree_id_of_parallel_ancestor_node_list)):
                    subtree_id = 0
                    for child in n.parallel_ancestor_node_list[i].children:
                        if subtree_id != n.sub_tree_id_of_parallel_ancestor_node_list[i]:
                            propagateParallelism(None, child, None, len(n.children))
                        subtree_id  = subtree_id + 1
        for child in n.children:
            to_visit.append(child)

    parallelism_feature_dict = collect_number_of_parallel_branches(tree)
    return parallelism_feature_dict


def propagateParallelism(parallel_ancestor_node_list, process_tree, sub_tree_id_of_parallel_ancestor_node_list, number_of_parallel_branches):
    if len(process_tree.parallel_ancestor_node_list) == 0:
        process_tree.parallel_ancestor_node_list = parallel_ancestor_node_list
        process_tree.number_of_parallel_branches = number_of_parallel_branches
        process_tree.sub_tree_id_of_parallel_ancestor_node_list = sub_tree_id_of_parallel_ancestor_node_list
    elif parallel_ancestor_node_list is None:
        process_tree.number_of_parallel_branches = process_tree.number_of_parallel_branches + number_of_parallel_branches - 1
    else:
        process_tree.parallel_ancestor_node_list = parallel_ancestor_node_list
        process_tree.sub_tree_id_of_parallel_ancestor_node_list = sub_tree_id_of_parallel_ancestor_node_list
        process_tree.number_of_parallel_branches = process_tree.number_of_parallel_branches + number_of_parallel_branches - 1
    for child in process_tree.children:
        propagateParallelism(parallel_ancestor_node_list, child, sub_tree_id_of_parallel_ancestor_node_list, number_of_parallel_branches)

#def propagateParallelismOnSubTreesOfAncestors(process_tree, sub_tree_id_of_parallel_ancestor_node_list):


def collect_number_of_parallel_branches(process_tree):
    parallelism_feature_dict = dict()
    to_visit = [process_tree]
    while len(to_visit) > 0:
        n = to_visit.pop(0)
        for child in n.children:
            to_visit.append(child)
        if n.operator is None and n.label is not None:
            if n.number_of_parallel_branches is None:
                parallelism_feature_dict[n.label] = 0
            else:
                parallelism_feature_dict[n.label] = 1 - (1 / n.number_of_parallel_branches)
    return parallelism_feature_dict


def new_parallelism_pathlength(process_tree):
    to_visit = [process_tree]
    parallelism_pathlength_dict = dict()
    while len(to_visit) > 0:
        n = to_visit.pop(0)
        if n.operator is pt_opt.Operator.PARALLEL:
            sub_trace_list = []
            for child in n.children:
                sub_trace_list.append(get_sub_traces(child))
            parallel_subtrace_list = combine_sublists_sequentially(sub_trace_list)
            # Initialize variables
            element_max_indices = {}
            longest_list_length = max(len(sublist) for sublist in parallel_subtrace_list)

            # Find the largest index for each element across all lists
            for i, sublist in enumerate(parallel_subtrace_list):
                for index, element in enumerate(sublist):
                    # Update the largest index where the element is found
                    element_max_indices[element] = max(element_max_indices.get(element, -1), index)

            # Calculate the values and store them in the dictionary
            for element, max_index in element_max_indices.items():
                parallelism_pathlength_dict[element] = max_index / longest_list_length


        else:
            for child in n.children:
                to_visit.append(child)
    return parallelism_pathlength_dict


def get_sub_traces(process_tree):
    sub_trace_list = []
    if process_tree.operator is pt_opt.Operator.XOR:
        for child in process_tree.children:
            sub_trace_list.extend(get_sub_traces(child))
        return sub_trace_list
    elif process_tree.operator is pt_opt.Operator.SEQUENCE:
        sequential_trace_list = []
        for child in process_tree.children:
            sequential_trace_list.append(get_sub_traces(child))
        sub_trace_list = combine_sublists_sequentially(sequential_trace_list)
        return sub_trace_list
    elif process_tree.operator is pt_opt.Operator.PARALLEL:
        parallel_trace_list = []
        for child in process_tree.children:
            parallel_trace_list.append(get_sub_traces(child))
        sub_trace_list = combine_sublists_sequentially(parallel_trace_list)
        return sub_trace_list
    elif process_tree.operator is pt_opt.Operator.LOOP:
        loop_trace_list = []
        loop_trace_list.extend(get_sub_traces(process_tree.children[0]))
        return loop_trace_list
    elif process_tree.operator is None and process_tree.label is not None:
        return [[process_tree.label]]
    elif process_tree.operator is None and process_tree.label is None:
        return []






#'''


def flatten_combination(combination):
    """
    Flatten a combination of lists, ensuring each element is treated as a list.
    """
    flattened = []
    for item in combination:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)  # Treat non-list elements as single items
    return flattened


def combine_sublists_sequentially(input_list):
    """
    Combine each sublist sequentially to generate all possible combinations.

    :param input_list: List of lists of lists to process.
    :return: A list of combined sublists.
    """
    # For each group, take all sublists as they are (no inter-group mixing)
    groups = [group for group in input_list]

    # Generate all combinations of one sublist from each group
    all_combinations = product(*groups)

    # Flatten the chosen sublists for each combination
    result = []
    for combination in all_combinations:
        combined = []
        for sublist in combination:
            combined.extend(sublist)
        result.append(combined)

    return result


def shuffle_two_sequences_preserve_order(seq1, seq2):
    """
    Perform the shuffle operation for two sequences, preserving the internal order of each sequence.

    :param seq1: The first sequence.
    :param seq2: The second sequence.
    :return: A list of all interleaved sequences, preserving the order of each input sequence.
    """
    if not seq1:
        return [seq2]
    if not seq2:
        return [seq1]

    # Recursive shuffle: place the first element of seq1 or seq2 at the front
    result = []
    for rest in shuffle_two_sequences_preserve_order(seq1[1:], seq2):
        result.append([seq1[0]] + rest)
    for rest in shuffle_two_sequences_preserve_order(seq1, seq2[1:]):
        result.append([seq2[0]] + rest)
    return result


def shuffle_sets_of_sequences_preserve_order(sets_of_sequences):
    """
    Shuffle multiple sets of sequences recursively, preserving the internal order of each sequence.

    :param sets_of_sequences: A list of lists of sequences (e.g., [[[2, 4, 5]], [[0]], [[7, 8, 9]]]).
    :return: A list of all possible interleaved traces, preserving internal order of each sequence.
    """
    if len(sets_of_sequences) == 1:
        # Base case: Only one set of sequences, no interleaving needed
        return sets_of_sequences[0]

    # Recursive case: Shuffle the first set with the result of the rest
    first_set = sets_of_sequences[0]
    rest_shuffled = shuffle_sets_of_sequences_preserve_order(sets_of_sequences[1:])

    # Interleave each sequence in the first set with each sequence in the shuffled rest
    result = []
    for seq1 in first_set:
        for seq2 in rest_shuffled:
            result.extend(shuffle_two_sequences_preserve_order(seq1, seq2))
    return result


def get_all_parallel_traces_preserve_order(input_list):
    """
    Generate all possible traces for a parallel node in a process tree, preserving order within each sequence.

    :param input_list: A list of lists of lists, where each sublist represents a set of sequences.
    :return: A list of all possible interleaved traces, preserving internal order of each sequence.
    """
    # For each group, take all sublists as they are (no flattening within groups)
    sets_of_sequences = [[sublist for sublist in group] for group in input_list]

    # Compute all possible interleavings
    return shuffle_sets_of_sequences_preserve_order(sets_of_sequences)


#input_list = [[[2, 4, 5], [3, 6]], [[0]], [[7, 8, 9], [10]]]

#output = combine_sublists_sequentially(input_list)

# Get all possible parallel traces, preserving order within each sequence
#parallel_traces = get_all_parallel_traces_preserve_order(input_list)
#print(parallel_traces)

'''
    if GenerationTree.operator is pt_opt.Operator.XOR:
        GenerationTree, activityStack = calculateFreqOfXOR(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.SEQUENCE:
        GenerationTree, activityStack = calculateFreqOfSequence(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.PARALLEL:
        GenerationTree, activityStack = calculateFreqOfParallel(GenerationTree, activityStack, lookUpTable)
    elif GenerationTree.operator is pt_opt.Operator.LOOP:
        GenerationTree, activityStack = calculateFreqOfLoop(GenerationTree, activityStack, lookUpTable)
    return GenerationTree, activityStack


'''
