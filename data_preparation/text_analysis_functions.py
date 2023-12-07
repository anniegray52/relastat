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

# from misc import *

from collections import Counter


def matrix_from_text(data, column_name, remove_stopwords=True, clean_text=True,
                     remove_email_addresses=False, update_stopwords=None,
                     **kwargs):
    """
    Create a matrix from a column of text data. 

    Parameters  
    ----------  
    data : pandas.DataFrame  
        The data to be used to create the matrix.   
    column_name : str   
        The name of the column containing the text data.
    remove_stopwords : bool 
        Whether to remove stopwords.
    clean_text : bool   
        Whether to clean the text data.     
    remove_email_addresses : bool   
        Whether to remove email addresses.  
    update_stopwords : list of str  
        The list of additional stopwords to be removed.    
    kwargs : dict   
        Other arguments to be passed to sklearn.feature_extraction.text.TfidfVectorizer.    

    Returns 
    ------- 
    Y : numpy.ndarray   
        The matrix created from the text data.  
    attributes : list of lists  
        The attributes of the nodes. The first list contains the attributes
        of the nodes in rows. The second list contains
        the attributes of the nodes in the columns. 
    """

    # gets rid of email addresses  in data
    if remove_email_addresses:
        data[column_name] = data.column_name.apply(
            lambda row: del_email_address(row))

    if clean_text:
        # gets rid of stopwords, symbols, makes lower case and base words
        data[column_name] = data.column_name.apply(
            lambda row: clean_text_(row))

    if remove_stopwords:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        if update_stopwords is not None:
            stopwords.update(update_stopwords)
        data[column_name] = data.column_name.apply(
            lambda row: remove_stopwords_(row, stopwords))

    # create tfidf matrix
    vectorizer = TfidfVectorizer(**kwargs)
    Y = vectorizer.fit_transform(data.column_name)
    attr0 = [{'name': i} for i in vectorizer.get_feature_names_out()]
    attr1 = [{'name': 'document_' + str(i)} for i in list(data.index)]
    attributes = [attr0, attr1]
    return Y, attributes


def del_email_address(text):
    """
    Not used by user."""
    e = '\S*@\S*\s?'
    pattern = re.compile(e)
    return pattern.sub('', text)


def clean_text_(text):
    """
    Not used by user."""
    return " ".join([Word(word).lemmatize() for word in re.sub("[^A-Za-z0-9]+", " ", text).lower().split()])


def remove_stopwords_(text, stopwords):
    """
    Not used by user."""
    return " ".join([word for word in text.split() if word not in stopwords])
