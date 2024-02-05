from scipy.stats import kendalltau
from scipy.cluster.hierarchy import dendrogram

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

def find_ancestors(model, target):
    """
    Find the ancestors of a target node in the dendrogram.  

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.   
    target : int    
        The target node.

    Returns 
    ------- 
    list    
        The list of ancestors of the target node.
    """
    n_samples = len(model.labels_)
    global ances
    for ind, merge in enumerate(model.children_):
        if target in merge:
            if n_samples+ind in ances:
                return [target] + ances[n_samples+ind]
            ances[n_samples+ind] = find_ancestors(model, n_samples+ind)
            return [target]+ances[n_samples+ind]
    return [ind+n_samples]


def find_descendents(model, node):
    """ 
    Find the descendents of a node in the dendrogram.   

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.   
    node : int  
        The target node.

    Returns 
    ------- 
    list    
        The list of descendents of the target node.
    """
    n_samples = len(model.labels_)
    global desc
    if node in desc:
        return desc[node]
    if node < n_samples:
        return [node]
    pair = model.children_[node-n_samples]
    desc[node] = find_descendents(
        model, pair[0])+find_descendents(model, pair[1])
    return desc[node]


def get_ranking(model, target):
    """ 
    Get the ranking of order of merges to other nodes.

    Parameters  
    ----------  
    model : AgglomerativeClustering 
        The fitted model.   
    target : int    
        The target node.

    Returns 
    ------- 
    list    
        The list of rankings of the other nodes.
    """
    rank = np.zeros(len(model.labels_))
    to_root = [find_descendents(model, cl)
               for cl in find_ancestors(model, target)]
    to_rank = [list(set(to_root[i+1]) - set(to_root[i]))
               for i in range(len(to_root)-1)]
    for i in range(1, len(to_rank)+1):
        rank[to_rank[i-1]] = i
    return rank


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

    global ances
    global desc

    ances = {}
    desc = {}
    ranking = np.array([get_ranking(model, t) for t in range(n)])
    kt = [kendalltau(ranking[i], true_ranking[i]
                     ).correlation for i in range(ranking.shape[0])]
    return np.mean(kt)
