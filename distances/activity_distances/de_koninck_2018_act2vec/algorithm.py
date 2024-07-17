from itertools import combinations

import gensim
import numpy as np
from sklearn.manifold import TSNE  # final reduction


def get_act2vec_distance_matrix(log, alphabet, sg = 0):
    vectorsize = 16
    #model = gensim.models.Word2Vec(sentences=log, vector_size=vectorsize, window=9, min_count=0, alpha=0.025, min_alpha=0.005, epochs=10, sg=sg)

    model = gensim.models.Word2Vec(sentences=log, vector_size=vectorsize, window=3, min_count=0, alpha=0.3,
                                   min_alpha=0.025, epochs=10, sg=sg)

    ''' 
    if random.randint(1,100) == 1:
        x_vals, y_vals, labels = reduce_dimensions(model)
        plot_function = plot_with_matplotlib

        plot_function(x_vals, y_vals, labels, len(alphabet))
    '''

    # Compute distances between all pairs of activities
    distances = {}
    for activity1, activity2 in combinations(alphabet, 2):
        distance = model.wv.distance(activity1, activity2)
        distances[(activity1, activity2)] = distance

    return distances


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels, number_of_activities):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, number_of_activities)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.show()
