Simple graph - Toy data 
=========================

.. code:: ipython3

    import relastat as rs
    import pandas as pd

.. code:: ipython3

    df = pd.DataFrame(
        {'A': ['a1', 'a1', 'a2', 'a2', 'a1', 'a1', 'a2', 'a2'],
            'B': ['b1', 'b2', 'b1', 'b2', 'b1', 'b2', 'b1', 'b2'],
            'ID': [1, 1, 1, 1, 2, 2, 2, 2]})
    relationships = ['A', 'B']
    time_col = 'ID'
    A1, attributes1 = rs.matrix_from_tables(
        df, relationships, dynamic_col=time_col, join_token='::')
    
    c0, att0 = rs.find_cc_containing_most(A1, attributes1, 'B', dynamic=False)
    c1, att1 = rs.find_cc_containing_most(A1, attributes1, 'A', dynamic=True)


.. parsed-literal::

    ['A', 'B']
    Number of connected components: 2
    Number of connected components: 2
