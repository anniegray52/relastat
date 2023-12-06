"""
Graph Functions
---------------
"""


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

from rela_py.data_preparation.misc_functions import *

from collections import Counter


def matrix_from_tables(tables, relationships, dynamic_col=None, weight_col=None, join_token='::'):
    """ 
    Create a DMP graph from a dataframe.    

    Parameters
    ----------
    tables : pandas.DataFrame or list of pandas.DataFrame
        The data to be used to create the graph. If a list of dataframes is 
        passed, each dataframe is used to create a separate graph.
    relationships : list of lists
        The partition pairs to be used to create the graph. Each element of
        the list is a list of two elements, which are the names of the
        partitions to be joined.
    dynamic_col : str or list of str
        The name of the column containing the time information. If a list of
        strings is passed, each string is the name of the column containing
        the time information for each dataframe in data.
    join_token : str
        The token used to join the partition name and the node name.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.  
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes that do not change over time. The second list contains
        the attributes of the nodes that change over time.
    """
    # Ensure data and relationships are in list format
    # add valueerror if not in the correct format
    if not isinstance(tables, list):
        tables = [tables]
    if isinstance(relationships[0], str):
        relationships = [relationships]
    if not isinstance(relationships[0][0], list):
        relationships = [relationships] * len(tables)
    # Handle the case when dynamic_col is None
    if dynamic_col is None:
        dynamic_col = [None] * len(tables)
    elif isinstance(dynamic_col, str):
        dynamic_col = [dynamic_col] * len(tables)
    if len(dynamic_col) != len(tables):
        dynamic_col = dynamic_col * len(tables)
    # Handle the case when weight_col is None
    if weight_col is None:
        weight_col = [None] * len(tables)
    elif isinstance(weight_col, str):
        weight_col = [weight_col] * len(tables)
    if len(weight_col) != len(tables):
        weight_col = weight_col * len(tables)

    edge_list = create_edge_list(
        tables, relationships, dynamic_col, join_token, weight_col)
    nodes, partitions, times, node_ids, time_ids = extract_node_time_info(
        edge_list, join_token)

    edge_list = transform_edge_data(edge_list, node_ids, time_ids, len(nodes))
    A = create_adjacency_matrix(edge_list, len(nodes), len(times), weight_col)
    attributes = create_node_attributes(
        nodes, partitions, times, len(nodes), len(times))

    return A.tocsr(), attributes


def create_edge_list(tables, relationships, dynamic_col, join_token, weight_col):
    edge_list = []
    for data0, relationships0, dynamic_col0, weight_col0 in zip(tables, relationships, dynamic_col, weight_col):
        for partition_pair in relationships0:
            if set(partition_pair).issubset(data0.columns):

                cols = deepcopy(partition_pair)
                colnames = ['V1', 'V2']

                if dynamic_col0:
                    cols.append(dynamic_col0)
                if weight_col0:
                    cols.append(weight_col0)
                    colnames.append('W')

                pair_data = deepcopy(
                    data0[cols].drop_duplicates())
                if not dynamic_col0:
                    pair_data['T'] = np.nan
                colnames.append('T')

                pair_data.columns = colnames

                if len(list(set(pair_data['V1']) & set(pair_data['V2']))) != 0:
                    pair_data['V1'] = [
                        f"{partition_pair[0]}{join_token}{x}" for x in pair_data['V1']]
                    pair_data['V2'] = [
                        f"{partition_pair[0]}{join_token}{x}" for x in pair_data['V2']]
                    pair_data['P1'] = partition_pair[0]
                    pair_data['P2'] = partition_pair[1]

                    edge_list.append(pair_data)
                else:
                    pair_data['V1'] = [
                        f"{partition_pair[0]}{join_token}{x}" for x in pair_data['V1']]
                    pair_data['V2'] = [
                        f"{partition_pair[1]}{join_token}{x}" for x in pair_data['V2']]
                    pair_data['P1'] = partition_pair[0]
                    pair_data['P2'] = partition_pair[1]

                    edge_list.append(pair_data)
                print(partition_pair)
    return pd.concat(edge_list)


def extract_node_time_info(edge_list, join_token):
    nodes = sorted(set(edge_list['V1']).union(edge_list['V2']))
    partitions = [node.split(join_token)[0] for node in nodes]
    times = sorted(set(edge_list['T'].unique()))
    # times = sorted(set(edge_list['T']))
    node_ids = {node: idx for idx, node in enumerate(nodes)}
    time_ids = {time: idx for idx, time in enumerate(times)}
    return nodes, partitions, times, node_ids, time_ids


def transform_edge_data(edge_list, node_ids, time_ids, n_nodes):
    edge_list['V_ID1'] = edge_list['V1'].map(node_ids)
    edge_list['V_ID2'] = edge_list['V2'].map(node_ids)
    edge_list['T_ID'] = edge_list['T'].map(time_ids)
    edge_list['X_ID1'] = edge_list['T_ID'] * n_nodes + edge_list['V_ID1']
    edge_list['X_ID2'] = edge_list['T_ID'] * n_nodes + edge_list['V_ID2']
    return edge_list


def create_adjacency_matrix(edge_list, n_nodes, n_times, weight_col):
    row_indices = pd.concat([edge_list['V_ID1'], edge_list['V_ID2']])
    col_indices = pd.concat([edge_list['X_ID2'], edge_list['X_ID1']])
    if weight_col[0]:
        values = pd.concat([edge_list['W'], edge_list['W']])
    else:
        values = np.ones(2 * len(edge_list))
    return sparse.coo_matrix((values, (row_indices, col_indices)), shape=(n_nodes, n_nodes * n_times))


def create_node_attributes(nodes, partitions, times, n_nodes, n_times):
    time_attrs = np.repeat(times, n_nodes)
    attributes = [
        [{'name': name, 'partition': partition, 'time': np.nan}
            for name, partition in zip(nodes, partitions)],
        [{'name': name, 'partition': partition, 'time': time} for name, partition,
            time in zip(nodes * n_times, partitions * n_times, time_attrs)]
    ]
    return attributes


def find_subgraph(A, attributes, subgraph_attributes):
    """
    Find a subgraph of a multipartite graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the multipartite graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    subgraph_attributes : list of lists
        The attributes of the nodes of the wanted in the subgraph. The first list contains
        the attributes of the nodes wanted in the rows. The second
        list contains the attributes of the nodes wanted in the column.

    Returns
    -------
    subgraph_A : scipy.sparse.csr_matrix
        The adjacency matrix of the subgraph.
    subgraph_attributes : list of lists
        The attributes of the nodes of the subgraph. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """

    if not isinstance(subgraph_attributes[0], list):
        subgraph_attributes[0] = [subgraph_attributes[0]]

    if not isinstance(subgraph_attributes[1], list):
        subgraph_attributes[1] = [subgraph_attributes[1]]

    # find the indices of the rows with required attributes
    subgraph_node_indices_row = []
    for node_idx, node_attributes in enumerate(attributes[0]):
        for each_subgraph_attributes in subgraph_attributes[0]:
            matched = True
            for key, value in each_subgraph_attributes.items():
                if key not in node_attributes or node_attributes[key] != value:
                    matched = False
                    break
            if matched:
                subgraph_node_indices_row.append(node_idx)

    # find the indices of the columns with required attributes
    subgraph_node_indices_col = []
    for node_idx, node_attributes in enumerate(attributes[1]):
        for each_subgraph_attributes in subgraph_attributes[1]:
            matched = True
            for key, value in each_subgraph_attributes.items():
                if key not in node_attributes or node_attributes[key] != value:
                    matched = False
                    break
            if matched:
                subgraph_node_indices_col.append(node_idx)

    # create subgraph and subgraph attributes
    subgraph_A = A[np.ix_(subgraph_node_indices_row,
                          subgraph_node_indices_col)]
    subgraph_attributes = [[attributes[0][i] for i in subgraph_node_indices_row], [
        attributes[1][i] for i in subgraph_node_indices_col]]

    return subgraph_A, subgraph_attributes


def symmetric_dilation(M):
    """
    Dilate a matrix to a symmetric matrix.
    """
    m, n = M.shape
    D = sparse.vstack([sparse.hstack([zero_matrix(m), M]),
                      sparse.hstack([M.T, zero_matrix(n)])])
    return D


def subgraph_idx(A, attributes, idx0, idx1):
    """
    Find a subgraph of a multipartite graph by indices.
    """
    subgraph_A = A[np.ix_(idx0, idx1)]
    subgraph_attributes = [
        [attributes[0][i] for i in idx0],
        [attributes[1][i] for i in idx1]
    ]
    return subgraph_A, subgraph_attributes


def find_connected_components(A, attributes, n_components=None):
    """
    Find connected components of a multipartite graph.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    n_components : int
        The number of components to be found.

    Returns
    -------
    cc_As : list of scipy.sparse.csr_matrix
        The adjacency matrices of the connected components.
    cc_attributes : list of lists
        The attributes of the nodes of the connected components. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """

    A_dilation = symmetric_dilation(A)
    _, cc = connected_components(A_dilation)
    print(f"Number of connected components: {_}")
    cc = [cc[:A.shape[0]], cc[A.shape[0]:]]
    if n_components == None:
        # n_components = np.max(cc) + 1
        n_components = _
    cc_As = []
    cc_attributes = []
    if n_components == 1:
        cc_As = A
        cc_attributes = attributes
    else:
        for i in range(n_components):
            idx0 = np.where(cc[0] == i)[0]
            idx1 = np.where(cc[1] == i)[0]
            store_cc_A, store_cc_attributes = subgraph_idx(
                A, attributes, idx0, idx1)
            cc_As.append(store_cc_A)
            cc_attributes.append(store_cc_attributes)

    return cc_As, cc_attributes


def find_cc_containing_most(A, attributes, partition, dynamic=False):
    """ 
    Find the connected component containing the most nodes from a partition.    

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.
    attributes : list of lists
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    partition : str
        The partition to be searched.
    dynamic : bool
        Whether we want the connected component containing the most nodes from dynamic part or not.

    Returns
    -------
    cc_A : scipy.sparse.csr_matrix
        The adjacency matrix of the connected component.
    cc_attributes : list of lists
        The attributes of the nodes of the connected component. The first list contains
        the attributes of the nodes in the rows. The second
        list contains the attributes of the nodes in the columns.
    """
    cc_A, cc_attributes = find_connected_components(
        A, attributes)
    if dynamic == False:
        attrs = [att[0] for att in cc_attributes]
        counts = [count_based_on_keys(att, 'partition') for att in attrs]
        select_idx = np.argmax([c[partition] for c in counts])
        return cc_A[select_idx], cc_attributes[select_idx]
    if dynamic == True:
        attrs = [att[1] for att in cc_attributes]
        counts = [count_based_on_keys(att, 'partition') for att in attrs]
        select_idx = np.argmax([c[partition] for c in counts])
        return cc_A[select_idx], cc_attributes[select_idx]


def to_networkx(A, attributes, symmetric=None):
    """ 
    Convert a multipartite graph to a networkx graph.
    """
    if symmetric == None:
        if is_symmetric(A):
            symmetric = True
        else:
            symmetric = False
    if symmetric:
        G_nx = nx.Graph(A)
        nx.set_node_attributes(
            G_nx, {i: a for i, a in enumerate(attributes[0])})
        return G_nx
    else:
        n0 = len(attributes[0])
        n1 = len(attributes[1])
        G_nx = nx.Graph(symmetric_dilation(A))
        nx.set_node_attributes(
            G_nx, {i: a for i, a in enumerate(attributes[0])})
        nx.set_node_attributes(
            G_nx, {i + n0: a for i, a in enumerate(attributes[1])})
        nx.set_node_attributes(G_nx, {i: {'bipartite': 0} for i in range(n0)})
        nx.set_node_attributes(
            G_nx, {i + n0: {'bipartite': 1} for i in range(n1)})
        return G_nx
