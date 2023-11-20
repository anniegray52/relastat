import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds
import networkx as nx
from copy import deepcopy
from scipy import linalg
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
import warnings


## ==================== ## misc functions ## ==================== ##

def symmetric_dilation(M):
    """
    Dilate a matrix to a symmetric matrix.
    """
    m, n = M.shape
    D = sparse.vstack([sparse.hstack([zero_matrix(m), M]),
                      sparse.hstack([M.T, zero_matrix(n)])])
    return D


def zero_matrix(m, n=None):
    """
    Create a zero matrix.
    """
    if n == None:
        n = m
    M = sparse.coo_matrix(([], ([], [])), shape=(m, n))
    return M


def safe_inv_sqrt(a, tol=1e-12):
    """
    Compute the inverse square root of an array, ignoring division by zero.
    """
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


def to_laplacian(A, regulariser=0):
    """
    Convert an adjacency matrix to a Laplacian matrix.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix.
    regulariser : float
        The regulariser to be added to the degrees of the nodes.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        The Laplacian matrix.
    """

    left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
    right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    if regulariser == 'auto':
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
    left_degrees_inv_sqrt = safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(
        left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
    return L


## ==================== ## dimension selection functions ## ==================== ##


def dim_select(A, plot=True, plotrange=50):
    ## IAN - NEEDS UPDATING FOR DILATED MATRIX ##
    """ 
    Select the number of dimensions for A.
    Finding a changepoint using the likelihood profile (Zhu, Ghodsi; 2006).

    Parameters
    ----------  
    As : numpy.array
        The array of matrices.
    plot : bool
        Whether to plot the singular values and the likelihood profile.
    plotrange : int
        The range of dimensions to be plotted.

    Returns
    -------
    lq_best : int
        The number of dimensions selected.
    """
    if scipy.sparse.issparse(A):
        A = A.todense()

    # # Construct rectangular matrices
    # if len(As.shape) == 2:
    #     As = np.array([As[:,:]])

    # if len(As.shape) == 3:
    #     T = len(As)
    #     A = As[0,:,:]
    #     for t in range(1,T):
    #         A = np.block([A,As[t]])

    UA, SA, VAt = np.linalg.svd(A)

    # Compute likelihood profile
    n = len(SA)
    lq = np.zeros(n)
    lq[0] = 'nan'
    for q in range(1, n):
        theta_0 = np.mean(SA[:q])
        theta_1 = np.mean(SA[q:])
        sigma = np.sqrt(
            ((q-1)*np.var(SA[:q]) + (n-q-1)*np.var(SA[q:])) / (n-2))
        lq_0 = np.sum(np.log(stats.norm.pdf(SA[:q], theta_0, sigma)))
        lq_1 = np.sum(np.log(stats.norm.pdf(SA[q:], theta_1, sigma)))
        lq[q] = lq_0 + lq_1
    lq_best = np.nanargmax(lq)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.0))
        plt.subplots_adjust(hspace=0.3)

        axs[0].plot(range(plotrange), SA[:plotrange], '.-')
        axs[0].set_title('Singular values')
        axs[0].set_xlabel('Number of dimensions')
        axs[0].axvline(x=lq_best, ls='--', c='k')

        axs[1].plot(range(plotrange), lq[:plotrange], '.-')
        axs[1].set_title('Log likelihood')
        axs[1].set_xlabel('Number of dimensions')
        axs[1].axvline(x=lq_best, ls='--', c='k')

    return lq_best


def wasserstein_dim_select(Y, split=0.5, rmin=1, rmax=50):
    """ 
    Select the number of dimensions for Y using Wasserstein distances.
    """
    n = Y.shape[0]
    train = round(n * split)
    rtry = int(np.min((train, rmax)))
    if sparse.issparse(Y):
        Y = Y.todense()
    Ytrain = Y[:train, :]
    Ytest = Y[train:n, :]
    U, s, Vh = sparse.linalg.svds(Ytrain, k=rtry-1)
    idx = s.argsort()[::-1]
    s = s[idx]
    Vh = Vh[idx, :]
    ws = []
    for r in tqdm(range(rmin, rtry+1)):
        P = Vh.T[:, :r] @ Vh[:r, :]
        Yproj = Ytrain @ P.T
        n1 = Yproj.shape[0]
        n2 = Ytest.shape[0]
        M = ot.dist(Yproj, Ytest, metric='euclidean')
        W1 = ot.emd2(np.repeat(1/n1, n1), np.repeat(1/n2, n2), M)
        ws.append(W1)
    return ws

## ==================== ## embedding functions ## ==================== ##


def embed(A, d=10, matrix='adjacency', regulariser=0):
    """ 
    Embed a graph using the Laplacian or adjacency matrix.  

    Parameters  
    ----------  
    A : scipy.sparse.csr_matrix  
        The adjacency matrix of the graph.  
    d : int 
        The dimension of the embedding.
    matrix : str    
        The matrix to be used for embedding. Should be 'adjacency' or 'laplacian'.
    regulariser : float 
        The regulariser to be added to the degrees of the nodes (if matrix = 'laplacian' used).    

    Returns 
    ------- 
    left_embedding : numpy.ndarray 
        The left embedding of the graph.    
    right_embedding : numpy.ndarray 
        The right embedding of the graph.    
    """

    # Check if there is more than one connected component
    num_components = connected_components(
        symmetric_dilation(A), directed=False)[0]

    if num_components > 1:
        warnings.warn(
            'Warning: More than one connected component in the graph.')
    if matrix not in ['adjacency', 'laplacian']:
        raise ValueError(
            "Invalid matrix type. Use 'adjacency' or 'laplacian'.")

    if matrix == 'laplacian':
        L = to_laplacian(A, regulariser)
        u, s, vT = svds(L, d)
    else:
        u, s, vT = svds(A, d)

    o = np.argsort(s[::-1])
    left_embedding = u[:, o] @ np.diag(np.sqrt(s[o]))
    right_embedding = vT.T[:, o] @ np.diag(np.sqrt(s[o]))

    return left_embedding, right_embedding


def recover_subspaces(embedding, attributes):
    """
    Recover the subspaces for each partition from an embedding.

    Parameters
    ----------
    embedding : numpy.ndarray
        The embedding of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.

    Returns
    -------
    partition_embeddings : dict
        The embeddings of the partitions.
    partition_attributes : dict
        The attributes of the nodes in the partitions.
    """

    partitions = list(set([x['partition'] for x in attributes]))
    partition_embeddings = {}
    partition_attributes = {}
    for p in partitions:
        p_embedding, p_attributes = select(
            embedding, attributes, {'partition': p})
        Y = p_embedding
        u, s, vT = linalg.svd(Y, full_matrices=False)
        o = np.argsort(s[::-1])
        Y = Y @ vT.T[:, o]
        partition_embeddings[p] = Y
        partition_attributes[p] = p_attributes
    return partition_embeddings, partition_attributes


def select(embedding, attributes, select_attributes):
    """
    Select portion of embedding and attributes associated with a set of attributes.
    """
    if not isinstance(select_attributes, list):
        select_attributes = [select_attributes]
    which_nodes = list()
    for attributes_dict in select_attributes:
        for a, v in attributes_dict.items():
            if not isinstance(v, list):
                v = [v]
        which_nodes_by_attribute = [[i for i, y in enumerate(
            attributes) if y[a] in v] for a, v in attributes_dict.items()]
        which_nodes.append(list(set.intersection(
            *map(set, which_nodes_by_attribute))))
    which_nodes = list(set().union(*which_nodes))
    selected_X = embedding[which_nodes, :]
    selected_attributes = [attributes[i] for i in which_nodes]
    return selected_X, selected_attributes


def truncate(X, d):
    """
    Truncate an embedding to a lower dimension.
    """
    Y = X[:, :d]
    return Y


def degree_correction(X):
    """
    Perform degree correction.
    """
    tol = 1e-12
    Y = deepcopy(X)
    norms = np.linalg.norm(X, axis=1)
    idx = np.where(norms > tol)
    Y[idx] = X[idx] / (norms[idx, None])
    return Y
