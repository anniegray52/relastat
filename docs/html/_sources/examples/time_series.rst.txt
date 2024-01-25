Time series - Stocks
====================

For this to you’ll need to download the data and set your own data path.

.. code:: ipython3

    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3

    # path = ''

.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    import relastat as rs

.. code:: ipython3

    data = pd.read_csv(path + '/sp500_data.csv', sep=',')

.. code:: ipython3

    Y, attributes = rs.matrix_from_time_series(data, 'date')

.. code:: ipython3

    (n, p) = Y.shape

.. code:: ipython3

    Y




.. parsed-literal::

    array([[-1.09947356, -1.48955518, -0.6515604 , ..., -2.20470748,
             1.90858336,  0.47525663],
           [ 0.16260559,  0.59510885, -2.05268816, ..., -1.39452167,
             4.45271374, -1.84401406],
           [ 0.50901386, -0.23915824,  0.18915015, ..., -2.02202849,
             2.5883366 , -1.38168959],
           ...,
           [ 0.98037899, -0.19029762, -0.32791901, ..., -1.28513025,
             0.95013665, -0.74156719],
           [ 0.29704411,  0.17612785,  0.59045864, ..., -2.57681779,
             2.98359571,  2.43695645],
           [ 1.75101968,  0.45417894,  0.77197585, ..., -2.83260922,
             0.54933395,  1.21645255]])


