from pm4py.util.regex import SharedObj, get_new_char

def get_act_correspondence(activities, parameters=None):
    """
    Gets an encoding for each activity

    Parameters
    --------------
    activities
        Activities of the two languages
    parameters
        Parameters

    Returns
    -------------
    encoding
        Encoding into hex characters
    """
    if parameters is None:
        parameters = {}

    shared_obj = SharedObj()
    ret = {}
    for act in activities:
        get_new_char(act, shared_obj)
        ret[act] = shared_obj.mapping_dictio[act]

    return ret

enc1, enc2 = encode_two_languages(lang1, lang2, parameters=parameters)

    # transform everything into a numpy array
    first_histogram = np.array([x[1] for x in enc1])
    second_histogram = np.array([x[1] for x in enc2])

    # including a distance matrix that includes the distance between
    # the traces
    distance_matrix = []
    for x in enc1:
        distance_matrix.append([])
        for y in enc2:
            # calculates the (normalized) distance between the strings
            dist = distance_function(x[0], y[0])
            distance_matrix[-1].append(float(dist))

    distance_matrix = np.array(distance_matrix)


