Multipartite graph
==================

Demonstrating how to construct a graph with multiple partitions.

For this to you’ll need to download the data and set your own data path.

.. code:: ipython3

    path = '/home/ag16115/Documents/phd/codebase_data/'

The dataset provides details about the procurement process in Brazil.
Each row includes information about a tender, including:

-  **Tender:** Tender ID
-  **Period:** Time duration
-  **Buyer:** Funding entity
-  **Item:** Description of the tender
-  **Company:** Bidding company
-  **Bidder Win:** Indicates whether the bid was successful or not

Import data

.. code:: ipython3

    import pandas as pd
    import matplotlib.pyplot as plt
    import relastat as rs

In the ``relationships`` input to ``rs.matrix_from_tables()``, we input
which partitions we want relationships between.

Here, we specfiy links between, - Company – Tender - Company – Buyer -
Company – Item

.. code:: ipython3

    data = pd.read_csv(path + 'brazil/activity_data.csv', sep = '\t', on_bad_lines='skip')
    A, attributes = rs.matrix_from_tables(data, [['Company', 'Tender'], ['Company', 'Buyer'],['Company', 'Item']],dynamic_col = 'Period', join_token='::')


.. parsed-literal::

    /tmp/ipykernel_396116/2869677282.py:1: DtypeWarning: Columns (13,16) have mixed types. Specify dtype option on import or set low_memory=False.
      data = pd.read_csv(path + 'brazil/activity_data.csv', sep = '\t', on_bad_lines='skip')


.. parsed-literal::

    ['Company', 'Tender']
    ['Company', 'Buyer']
    ['Company', 'Item']


We can find a subgraph of A based on attributes using
``rs.find_subgraph``

.. code:: ipython3

    subgraph_attributes = [
        [{'partition': 'Company'},{'partition': 'Tender'}],
        {'partition': 'Buyer'}
    ]
    subgraph_A, subgraph_attributes  = rs.find_subgraph(A, attributes,subgraph_attributes)

.. code:: ipython3

    # A_dilation = symmetric_dilation(subgraph_A)
    # is_symmetric(A_dilation)s

Find the largest connected component of the graph

.. code:: ipython3

    cc_A, cc_attributes = rs.find_connected_components(A, attributes, n_components = 1)


.. parsed-literal::

    Number of connected components: 217748


Embedding

.. code:: ipython3

    d = 10
    embedding = rs.embed(A, d=d)


.. parsed-literal::

    /home/ag16115/.local/lib/python3.8/site-packages/relastat/embedding.py:202: UserWarning: Warning: More than one connected component in the graph.
      warnings.warn(

