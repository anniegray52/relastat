import networkx as nx
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.stats import kendalltau
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ========== Plot dendrogram ==========


def plot_dendrogram(model, **kwargs):
    """
    Create linkage matrix and then plot the dendrogram  

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model to plot.
    **kwargs : dict 
        Keyword arguments for dendrogram function.

    Returns 
    ------- 
    None
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# ========== Rankings ==========


def get_ranking(model):
    lm = linkage_matrix(model, rescale=False)
    cophenetic_dists = squareform(cophenet(lm))
    mh_rank = np.array([rankdata(cophenetic_dists[i], method='dense')
                       for i in range(cophenetic_dists.shape[0])])
    mh_rank = mh_rank - 1
    return mh_rank


def kendalltau_similarity(model, true_ranking):
    """ 
    Calculate the Kendall's tau similarity between the model and true ranking.  

    NEEDS TESTING (does global work?)    

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.   
    true_ranking : array-like, shape (n_samples, n_samples) 
        The true ranking of the samples.

    Returns 
    ------- 
    float   
        The mean Kendall's tau similarity between the model and true ranking.
    """

    if model.labels_.shape[0] != true_ranking.shape[0]:
        raise ValueError(
            "The number of samples in the model and true_ranking must be the same.")
    n = model.labels_.shape[0]

    ranking = get_ranking(model)
    kt = [kendalltau(ranking[i], true_ranking[i]
                     ).correlation for i in tqdm(range(ranking.shape[0]))]
    return np.mean(kt)


def varimax(Phi, gamma=1, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p)
                                        * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d/d_old < tol:
            break
    return np.dot(Phi, R)


def plot_HC_clustering(model, node_colours=None, no_merges=None, labels=None, plot_labels=None, internal_node_colour='black',
                       linewidths=None, edgecolors=None, leaf_node_size=20, fontsize=10, internal_node_size=1, figsize=(10, 10)):

    if no_merges is None:
        no_merges = model.children_.shape[0]
    if node_colours is None:
        node_colours = np.repeat('skyblue', model.n_leaves_)
    if labels is None:
        labels = np.repeat('', model.children_.shape[0] + model.n_leaves_)

    data = model.children_[:no_merges, :]
    n = model.n_leaves_

    G = nx.Graph()

    for i in range(data.shape[0]):
        idx = i + n
        to_merge = data[i]
        G.add_edge(to_merge[0], idx)
        G.add_edge(to_merge[1], idx)

    node_colours_ = {}
    node_sizes = {}
    node_names = {}
    for node in G.nodes():
        if node < n:
            node_colours_[node] = node_colours[node]
            node_sizes[node] = leaf_node_size
            node_names[node] = labels[node]
        else:
            node_colours_[node] = internal_node_colour
            node_sizes[node] = internal_node_size
            node_names[node] = ''

    # Draw the graph with node sizes and names
    plt.figure(figsize=figsize)  # Adjust figure size

    positions = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
    nx.draw(G, positions, with_labels=False, node_size=[
            node_sizes[node] for node in G.nodes()], node_color=[
            node_colours_[node] for node in G.nodes()], edgecolors=edgecolors, linewidths=linewidths)

    nx.draw_networkx_labels(
        G, positions, labels=node_names, font_size=fontsize)
    plt.show()


def linkage_matrix(model, rescale=False):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    if rescale == True:
        d_max = np.max(model.distances_)
        d_min = np.min(model.distances_)
        distances = (model.distances_ - d_min) / (d_max - d_min)
    else:
        distances = model.distances_

    linkage_matrix = np.column_stack(
        [model.children_, distances, counts]
    ).astype(float)

    return linkage_matrix


def get_triu(matrix, k=1):
    return matrix[np.triu_indices(matrix.shape[0], k=k)]


def utri2mat(utri):
    n = int(-1 + np.sqrt(1 + 8*len(utri))) // 2
    iu1 = np.triu_indices(n)
    ret = np.empty((n, n))
    ret[iu1] = utri
    ret.T[iu1] = utri
    return ret


def sample_hyperbolicity(data, metric='dot_products', distance=False, num_samples=5000):

    valid_distances = ['dot_products', 'cosine_similarity', 'precomputed', 'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard',
                       'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine', 'kulsinski']

    if metric not in valid_distances:
        raise ValueError(
            'Invalid distance metric, please choose from {}'.format(valid_distances))

    if metric == 'dot_products':
        initial_mhs = data @ data.T
    if metric == 'cosine_similarity':
        initial_mhs = cosine_similarity(data)
    if metric == 'precomputed':
        initial_mhs = data
    if metric != 'dot_products' and metric != 'cosine_similarity' and metric != 'precomputed':
        initial_mhs = - pairwise_distances(data, metric=metric)
        # initial_mhs = np.max(initial_mhs) - initial_mhs

    if distance == False:
        print('Calculating distance matrix')
        ranks = rankdata(get_triu(initial_mhs, k=0), method='average')
        c_ranks = ranks / np.max(ranks)
        mhs = utri2mat(c_ranks)
        # heights = np.repeat(np.diag(mhs), n).reshape((n, n))
        heights = np.repeat(1, n**2).reshape((n, n))
        print(heights)
        distance_matrix = heights + heights.T - 2*mhs
    if distance == True:
        print('Calculating distance matrix')
        # ranks = rankdata(
        #     [-1 * j for j in get_triu(initial_mhs, k=0)], method='average')
        ranks = rankdata(get_triu(initial_mhs, k=0), method='average')
        c_ranks = ranks / np.max(ranks)
        mhs = utri2mat(c_ranks)
        heights = np.repeat(np.diag(mhs), n).reshape((n, n))
        distance_matrix = heights + heights.T - 2*mhs
        print(np.min(distance_matrix), np.max(distance_matrix))

    hyps = []
    print('Calclating hyperbolicity')
    for i in tqdm(range(num_samples)):
        node_tuple = np.random.choice(range(n), 4, replace=False)
        try:
            d01 = distance_matrix[node_tuple[0], node_tuple[1]]
            d23 = distance_matrix[node_tuple[2], node_tuple[3]]
            d02 = distance_matrix[node_tuple[0], node_tuple[2]]
            d13 = distance_matrix[node_tuple[1], node_tuple[3]]
            d03 = distance_matrix[node_tuple[0], node_tuple[3]]
            d12 = distance_matrix[node_tuple[1], node_tuple[2]]

            s = [d01 + d23, d02 + d13, d03 + d12]
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    # return hyps
    return np.max(hyps), np.mean(hyps)
