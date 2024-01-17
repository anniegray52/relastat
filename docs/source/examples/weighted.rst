Weighted graph - Harry Potter
=============================

NEEDS DOING - haven't finished.

.. code:: ipython3

    import pandas as pd


.. code:: ipython3

    import relastat as rs

.. code:: ipython3

    # replace with own path to data folder:
    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3

    data = pd.read_csv(path + '/harry_potter.csv', sep=',')
    attributes = pd.read_csv(path + '/HP-characters.csv', sep=',')

.. code:: ipython3

    ## replace - in type column to -1 and + to 1    
    data['type'] = [1 if t == '+' else -1 for t in list(data['type'])]

.. code:: ipython3

    A, attributes = rs.matrix_from_tables(data, ['source','target'], weight_col=None)


.. parsed-literal::

    ['source', 'target']