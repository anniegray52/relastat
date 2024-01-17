Dynamic graph - Lyon school data
================================

.. code:: ipython3

    import pandas as pd
    from scipy import sparse
    from scipy.sparse.linalg import svds
    from scipy import linalg
    import matplotlib.pyplot as plt


.. code:: ipython3

    import relastat as rs


.. code:: ipython3

    # replace with own path to data folder:
    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3

    # load data 
    data = pd.read_csv(path + 'ia-primary-school-proximity-attr.edges', sep = ',', on_bad_lines='skip', header = None)

.. code:: ipython3

    ## rename columns   
    data.columns = ['V1', 'V2', 'T', 'L1', 'L2']
    ## sort out time column
    data['H'] = [int(int(t)/(60*60)) for t in list(data['T'])]
    data['D'] = [int(int(t)/(60*60*24)) for t in list(data['T'])]
    data['T1'] = [10*int(i/24) + i%24 - 8 for i in list(data['H'])]

.. code:: ipython3

    def scree_plot(A, s = 1, vline=None):
        UA, SA, VAt = scipy.sparse.linalg.svds(A,k=50)
        SA = SA[::-1]

.. code:: ipython3

    def scree_plot(A, k = 50, s = 10, vline=None):
        UA, SA, VAt = scipy.sparse.linalg.svds(A,k=k)
        fig=plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
        plt.scatter(range(len(SA)), np.sort(SA)[::-1], s =s)
        if vline:
            plt.axvline(x=vline, color='green', linewidth=0.5)

.. code:: ipython3

    A, attributes = rs.matrix_from_tables(data, ['V1','V2'], dynamic_col='T', join_token='::')


.. parsed-literal::

    ['V1', 'V2']


.. code:: ipython3

    d = 10
    scree_plot(A, k = 20,s = d, vline=10)



.. image:: tutorial_files/tutorial_14_0.png


.. code:: ipython3

    d = 10
    embedding = rs.embed(A, d = d, right_embedding=True)
    LHS = embedding[0]  
    RHS = embedding[1]  
    
    LHS = rs.degree_correction(LHS)      
    RHS = rs.degree_correction(RHS)  



.. parsed-literal::

    /home/ag16115/Documents/phd/codebase/relastat/embedding.py:180: UserWarning: Warning: More than one connected component in the graph.
      warnings.warn(


.. code:: ipython3

    ## plot the right hand side embedding and colour by time    
    fig=plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')   
    plt.scatter(RHS[:,0], RHS[:,1], c = [att['time'] for att in attributes[1]], s = 1, cmap = 'viridis')    
    plt.colorbar()
    plt.show()



.. image:: tutorial_files/tutorial_16_0.png
