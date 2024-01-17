Time series - Stocks
=====================
.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    import relastat as rs


.. code:: ipython3

    # replace with own path to data folder:
    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3

    data = pd.read_csv(path + '/sp500_data.csv', sep=',')

.. code:: ipython3

    Y, attributes = rs.matrix_from_time_series(data, 'date')

.. code:: ipython3

    (n,p) = Y.shape