import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds
import networkx as nx
from copy import deepcopy
from scipy import linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
# import ot

from relastat.utils import zero_matrix, symmetric_dilation, safe_inv_sqrt

## ==================== ## misc functions ## ==================== ##


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
    Finding a changepoint using the likelihood profile (Zhu, Ghodsi; 2006). I've just lifted this from Ian so it might need updating.

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


# def wasserstein_dim_select(Y, split=0.5, rmin=1, rmax=50):
#     """
#     Select the number of dimensions for Y using Wasserstein distances.

#     Parameters
#     ----------
#     Y : numpy.ndarray
#         The array of matrices.
#     split : float
#         The proportion of the data to be used for training.
#     rmin : int
#         The minimum number of dimensions to be considered.
#     rmax : int
#         The maximum number of dimensions to be considered.

#     Returns
#     -------
#     ws : list of numpy.ndarray
#         The Wasserstein distances between the training and test data for each number of dimensions.
#     """

#     try:
#         import ot
#     except ModuleNotFoundError:
#         logging.error("ot not found, pip install pot")
#     print('tensorflow warnings are seemingly a bug in ot, ignore them')
#     n = Y.shape[0]
#     train = round(n * split)
#     rtry = int(np.min((train, rmax)))
#     if sparse.issparse(Y):
#         Y = Y.todense()
#     Ytrain = Y[:train, :]
#     Ytest = Y[train:n, :]
#     U, s, Vh = svds(Ytrain, k=rtry-1)
#     idx = s.argsort()[::-1]
#     s = s[idx]
#     Vh = Vh[idx, :]
#     ws = []
#     for r in tqdm(range(rmin, rtry+1)):
#         P = Vh.T[:, :r] @ Vh[:r, :]
#         Yproj = Ytrain @ P.T
#         n1 = Yproj.shape[0]
#         n2 = Ytest.shape[0]
#         M = ot.dist(Yproj, Ytest, metric='euclidean')
#         W1 = ot.emd2(np.repeat(1/n1, n1), np.repeat(1/n2, n2), M)
#         ws.append(W1)
#     return ws


def WassersteinDimensionSelect(Y, dims, split=0.5):
    """ 
    Select the number of dimensions for Y using Wasserstein distances.  

    Parameters  
    ----------  
    Y : numpy.ndarray   
        The array of matrices.  
    dims : list of int  
        The dimensions to be considered.    
    split : float   
        The proportion of the data to be used for training. 

    Returns 
    ------- 
    ws : list of numpy.ndarray   
        The Wasserstein distances between the training and test data for each number of dimensions.    
    """

    try:
        import ot
    except ModuleNotFoundError:
        logging.error("ot not found, pip install pot")
    print('tensorflow warnings are seemingly a bug in ot, ignore them')

    n = Y.shape[0]
    idx = np.random.choice(range(n), int(n*split), replace=False)
    Y1 = Y[idx]

    mask = np.ones(Y.shape[0], dtype=bool)
    mask[idx] = False
    Y2 = Y[mask]

    if sparse.issparse(Y2):
        Y2 = Y2.toarray()
    n1 = Y1.shape[0]
    n2 = Y2.shape[0]
    max_dim = np.max(dims)
    U, S, Vt = sparse.linalg.svds(Y1, k=max_dim)
    S = np.flip(S)
    Vt = np.flip(Vt, axis=0)
    Ws = []
    for dim in dims:
        M = ot.dist((Y1 @ Vt.T[:, :dim]) @ Vt[:dim, :],
                    Y2, metric='euclidean')
        Ws.append(ot.emd2(np.repeat(1/n1, n1), np.repeat(1/n2, n2), M))
        print(
            f'Number of dimensions: {dim:d}, Wasserstein distance {Ws[-1]:.5f}')
    return Ws

## ==================== ## embedding functions ## ==================== ##


# def embed(Y, d=10, right_embedding=False, make_laplacian=False, regulariser=0):
#     ## FIX - when to sqrt the singular values? ##
#     """
#     Embed a matrix.

#     Parameters
#     ----------
#     Y : numpy.ndarray
#         The array of matrices.
#     d : int
#         The number of dimensions to embed into.
#     right_embedding : bool
#         Whether to return the right embedding.
#     make_laplacian : bool
#         Whether to use the Laplacian matrix.
#     regulariser : float
#         The regulariser to be added to the degrees of the nodes. (only used if make_laplacian=True)

#     Returns
#     -------
#     left_embedding : numpy.ndarray
#         The left embedding.
#     right_embedding : numpy.ndarray
#         The right embedding. (only returned if right_embedding=True)
#     """

#     # Check if there is more than one connected component
#     num_components = connected_components(
#         symmetric_dilation(Y), directed=False)[0]

#     if num_components > 1:
#         warnings.warn(
#             'Warning: More than one connected component in the graph.')
#     # if matrix not in ['adjacency', 'laplacian']:
#     #     raise ValueError(
#     #         "Invalid matrix type. Use 'adjacency' or 'laplacian'.")

#     if make_laplacian == True:
#         L = to_laplacian(Y, regulariser)
#         u, s, vT = svds(L, d)
#     else:
#         u, s, vT = svds(Y, d)

#     o = np.argsort(s[::-1])
#     left_embedding = u[:, o] @ np.diag(np.sqrt(s[o]))

#     if right_embedding == True:
#         right_embedding = vT.T[:, o] @ np.diag(np.sqrt(s[o]))
#         return left_embedding, right_embedding
#     else:
#         return left_embedding


def embed(Y, d=10, version='sqrt', right_embedding=False, make_laplacian=False, regulariser=0):
    """ 
    Embed a matrix.   

    Parameters  
    ----------  
    Y : numpy.ndarray
        The array of matrices.  
    d : int 
        The number of dimensions to embed into. 
    version : str   
        The version of the embedding. Options are 'full' or 'sqrt' (default).   
    right_embedding : bool  
        Whether to return the right embedding.
    make_laplacian : bool   
        Whether to use the Laplacian matrix.
    regulariser : float 
        The regulariser to be added to the degrees of the nodes. (only used if make_laplacian=True) 

    Returns 
    ------- 
    left_embedding : numpy.ndarray
        The left embedding. 
    right_embedding : numpy.ndarray 
        The right embedding. (only returned if right_embedding=True)    
    """

    # Check if there is more than one connected component
    num_components = connected_components(
        symmetric_dilation(Y), directed=False)[0]

    if num_components > 1:
        warnings.warn(
            'Warning: More than one connected component in the graph.')

    if version not in ['full', 'sqrt']:
        raise ValueError('version must be full or sqrt (default)')

    if make_laplacian == True:
        L = to_laplacian(Y, regulariser)
        u, s, vT = svds(L, d)
    else:
        u, s, vT = svds(Y, d)

    if version == 'sqrt':
        o = np.argsort(s[::-1])
        S = np.sqrt(s[o])
    if version == 'full':
        o = np.argsort(s[::-1])
        S = s[o]

    o = np.argsort(s[::-1])
    left_embedding = u[:, o] @ np.diag(S)

    if right_embedding == True:
        right_embedding = vT.T[:, o] @ np.diag(S)
        return left_embedding, right_embedding
    else:
        return left_embedding


def eigen_decomp(A, dim=None):
    """ 
    Perform eigenvalue decomposition of a matrix.   

    Parameters  
    ----------  
    A : numpy.ndarray   
        The matrix to be decomposed.
    dim : int   
        The number of dimensions to be returned.    

    Returns 
    ------- 
    eigenvalues : numpy.ndarray 
        The eigenvalues.    
    eigenvectors : numpy.ndarray    
        The eigenvectors.   
    """

    eigenvalues, eigenvectors = np.linalg.eig(A)
    # find nonzero eigenvalues and their corresponding eigenvectors
    idx = np.where(np.round(eigenvalues, 4) != 0)[0]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if dim is not None:
        eigenvalues = eigenvalues[:dim]
        eigenvectors = eigenvectors[:, :dim]

    return eigenvalues, eigenvectors


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

    Parameters  
    ----------  
    embedding : numpy.ndarray   
        The embedding of the graph.
    attributes : list of lists  
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns. 
    select_attributes : dict or list of dicts   
        The attributes to select by. If a list of dicts is provided, the intersection of the nodes
        satisfying each dict is selected.

    Returns 
    ------- 
    selected_X : numpy.ndarray   
        The selected embedding. 
    selected_attributes : list of lists 
        The attributes of the selected nodes.
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


def degree_correction(X):
    """
    Perform degree correction.  

    Parameters  
    ----------  
    X : numpy.ndarray   
        The embedding of the graph. 

    Returns 
    ------- 
    Y : numpy.ndarray   
        The degree-corrected embedding. 
    """
    tol = 1e-12
    Y = deepcopy(X)
    norms = np.linalg.norm(X, axis=1)
    idx = np.where(norms > tol)
    Y[idx] = X[idx] / (norms[idx, None])
    return Y
