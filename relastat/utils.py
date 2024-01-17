
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

## ==================== ## misc functions ## ==================== ##


def safe_inv_sqrt(a, tol=1e-12):
    """
    Compute the inverse square root of an array, ignoring division by zero.
    """
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


def intersection(lst1, lst2):
    """Find the intersection of two lists"""
    return list(set(lst1) & set(lst2))


def union(lst1, lst2):
    """Find the union of two lists"""
    final_list = list(set(lst1) | set(lst2))
    return final_list


def zero_matrix(m, n=None):
    """
    Create a zero matrix.
    """
    if n == None:
        n = m
    M = sparse.coo_matrix(([], ([], [])), shape=(m, n))
    return M


def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        return False

    if not isinstance(m, sparse.coo_matrix):
        m = sparse.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def count_based_on_keys(list_of_dicts, selected_keys):
    if isinstance(selected_keys, str):
        counts = Counter(d[selected_keys] for d in list_of_dicts)
    elif len(selected_keys) == 1:
        counts = Counter(d[selected_keys[0]] for d in list_of_dicts)
    else:
        counts = Counter(tuple(d[key] for key in selected_keys)
                         for d in list_of_dicts)
    return counts


def symmetric_dilation(M):
    """
    Dilate a matrix to a symmetric matrix.
    """
    m, n = M.shape
    D = sparse.vstack([sparse.hstack([zero_matrix(m), M]),
                      sparse.hstack([M.T, zero_matrix(n)])])
    return D


def truncate(X, d):
    """
    Truncate an embedding to a lower dimension.
    """
    Y = X[:, :d]
    return Y
