Multipartite graph - Public procurement data 
============================================

.. code:: ipython3

    import pandas as pd
    import matplotlib.pyplot as plt


.. code:: ipython3

    import relastat as rs

The data contains information about the procurement process in Brazil.
Each row contains information about a tender with information: 
- Tender: tender id 
- Period: time 
- Buyer: who is funding 
- Item: what the tender is about 
- Company: who has bid for the tender 
- bidder_win: whether the bid was won or not

.. code:: ipython3

    # replace with own path to data folder:
    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3

    data = pd.read_csv(path + 'brazil/activity_data.csv', sep = '\t', on_bad_lines='skip')
    A, attributes = rs.matrix_from_tables(data, [['Company', 'Tender'], ['Company', 'Buyer'],['Company', 'Item']],dynamic_col = 'Period', join_token='::')


.. parsed-literal::

    /tmp/ipykernel_50906/2869677282.py:1: DtypeWarning: Columns (13,16) have mixed types. Specify dtype option on import or set low_memory=False.
      data = pd.read_csv(path + 'brazil/activity_data.csv', sep = '\t', on_bad_lines='skip')


.. parsed-literal::

    ['Company', 'Tender']
    ['Company', 'Buyer']
    ['Company', 'Item']


.. code:: ipython3

    # find subgraph wanted
    
    subgraph_attributes = [
        [{'partition': 'Company'},{'partition': 'Tender'}],
        {'partition': 'Buyer'}
    ]
    
    # subgraph_attributes = [
    #     {'partition': 'Company'},
    #     {'partition': 'Buyer'}
    # ]
    subgraph_A, subgraph_attributes  = rs.find_subgraph(A, attributes,subgraph_attributes)

.. code:: ipython3

    # A_dilation = symmetric_dilation(subgraph_A)
    # is_symmetric(A_dilation)

.. code:: ipython3

    # take the largest connected component
    cc_A, cc_attributes = rs.find_connected_components(A, attributes,n_components = 1)


.. parsed-literal::

    Number of connected components: 217748


.. code:: ipython3

    d = 10
    embedding = rs.embed(A, d=d)


.. parsed-literal::

    /home/ag16115/Documents/phd/codebase/relastat/embedding.py:180: UserWarning: Warning: More than one connected component in the graph.
      warnings.warn(

