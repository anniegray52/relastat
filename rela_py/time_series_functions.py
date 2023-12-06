"""
Time Series Functions
----------------------
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


def matrix_from_time_series(data, time_col, drop_nas=True):
    """ 
    Create a matrix from a time series. 

    Parameters  
    ----------  
    data : pandas.DataFrame  
        The data to be used to create the matrix.   
    time_col : str  
        The name of the column containing the time information.
    drop_nas : bool 
        Whether to drop rows with missing values.

    Returns 
    ------- 
    Y : numpy.ndarray   
        The matrix created from the time series.
    attributes : list of lists  
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns.
    """
    data = data.sort_values(by=time_col)
    data = data.dropna(axis=1, how='any')

    times = list(data[time_col])
    data.drop([time_col], axis=1, inplace=True)
    ids = list(data.columns)

    Y = np.array(data).T
    attributes = [
        [{'name': i} for i in ids], [{'time': i} for i in times]
    ]
    return Y, attributes
