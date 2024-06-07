20Newsgroup data
================

Introduction
------------

In this notebook, we will demonstrate how to use the following functions
on the 20Newsgroup data - ``matrix_from_text()`` -
``WassersteinDimensionSelect()`` - ``embed()`` - ``plot_dendrogram()`` -
``plot_HC_clustering()``

Each document is associated with 1 of 20 newsgroup topics, organized at
two hierarchical levels.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    import relastat as rs

.. code:: ipython3

    # pip install git+ssh://git@github.com/anniegray52/relastat.git --upgrade

Data load
---------

Import data and create dataframe.

.. code:: ipython3

    newsgroups = fetch_20newsgroups() 
    
    df = pd.DataFrame()
    df["data"] = newsgroups["data"]
    df["target"] = newsgroups["target"]
    df["target_names"] = df.target.apply(
        lambda row: newsgroups["target_names"][row])
    df[['layer1', 'layer2']] = df['target_names'].str.split('.', n=1, expand=True)

.. code:: ipython3

    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>data</th>
          <th>target</th>
          <th>target_names</th>
          <th>layer1</th>
          <th>layer2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>From: lerxst@wam.umd.edu (where's my thing)\nS...</td>
          <td>7</td>
          <td>rec.autos</td>
          <td>rec</td>
          <td>autos</td>
        </tr>
        <tr>
          <th>1</th>
          <td>From: guykuo@carson.u.washington.edu (Guy Kuo)...</td>
          <td>4</td>
          <td>comp.sys.mac.hardware</td>
          <td>comp</td>
          <td>sys.mac.hardware</td>
        </tr>
        <tr>
          <th>2</th>
          <td>From: twillis@ec.ecn.purdue.edu (Thomas E Will...</td>
          <td>4</td>
          <td>comp.sys.mac.hardware</td>
          <td>comp</td>
          <td>sys.mac.hardware</td>
        </tr>
        <tr>
          <th>3</th>
          <td>From: jgreen@amber (Joe Green)\nSubject: Re: W...</td>
          <td>1</td>
          <td>comp.graphics</td>
          <td>comp</td>
          <td>graphics</td>
        </tr>
        <tr>
          <th>4</th>
          <td>From: jcm@head-cfa.harvard.edu (Jonathan McDow...</td>
          <td>14</td>
          <td>sci.space</td>
          <td>sci</td>
          <td>space</td>
        </tr>
      </tbody>
    </table>
    </div>



For a random sample of the data, create tf-idf features.

.. code:: ipython3

    n = 5000
    df = df.sample(n=n, replace=False, random_state=22).reset_index(drop=True)

\`rs.’matrix_from_text’ - creates a Y matrix of tf-idf features. It
takes in a dataframe and the column which contains the data. Further
functionality includes: removing general stopwords, adding stopwords,
removing email addresses, cleaning (lemmatize and remove symbol,
lowercase letters) and a threshold for the min/max number of documents a
word needs to appear in to be included.

.. code:: ipython3

    Y, attributes = rs.matrix_from_text(df, 'data', remove_stopwords=True, clean_text=True,
                                        remove_email_addresses=True, update_stopwords=['subject'],
                                        min_df=5, max_df=len(df)-1000)

.. code:: ipython3

    (n,p) = Y.shape
    print("n = {}, p = {}".format(n,p))


.. parsed-literal::

    n = 5000, p = 12804


Perform dimension selection using Wasserstein distances, see [^1] for
details

.. code:: ipython3

    ws = rs.WassersteinDimensionSelect(Y, range(40), split=0.5)
    dim = np.argmin(ws)


.. parsed-literal::

    2024-06-07 15:24:36.733201: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-06-07 15:24:36.777470: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-06-07 15:24:37.612544: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    tensorflow warnings are seemingly a bug in ot, ignore them
    Number of dimensions: 0, Wasserstein distance 1.00000
    Number of dimensions: 1, Wasserstein distance 0.98924
    Number of dimensions: 2, Wasserstein distance 0.98642
    Number of dimensions: 3, Wasserstein distance 0.98465
    Number of dimensions: 4, Wasserstein distance 0.98280
    Number of dimensions: 5, Wasserstein distance 0.98164
    Number of dimensions: 6, Wasserstein distance 0.98034
    Number of dimensions: 7, Wasserstein distance 0.97934
    Number of dimensions: 8, Wasserstein distance 0.97839
    Number of dimensions: 9, Wasserstein distance 0.97756
    Number of dimensions: 10, Wasserstein distance 0.97663
    Number of dimensions: 11, Wasserstein distance 0.97613
    Number of dimensions: 12, Wasserstein distance 0.97566
    Number of dimensions: 13, Wasserstein distance 0.97500
    Number of dimensions: 14, Wasserstein distance 0.97467
    Number of dimensions: 15, Wasserstein distance 0.97428
    Number of dimensions: 16, Wasserstein distance 0.97421
    Number of dimensions: 17, Wasserstein distance 0.97405
    Number of dimensions: 18, Wasserstein distance 0.97385
    Number of dimensions: 19, Wasserstein distance 0.97376
    Number of dimensions: 20, Wasserstein distance 0.97353
    Number of dimensions: 21, Wasserstein distance 0.97347
    Number of dimensions: 22, Wasserstein distance 0.97325
    Number of dimensions: 23, Wasserstein distance 0.97321
    Number of dimensions: 24, Wasserstein distance 0.97311
    Number of dimensions: 25, Wasserstein distance 0.97314
    Number of dimensions: 26, Wasserstein distance 0.97312
    Number of dimensions: 27, Wasserstein distance 0.97310
    Number of dimensions: 28, Wasserstein distance 0.97314
    Number of dimensions: 29, Wasserstein distance 0.97315
    Number of dimensions: 30, Wasserstein distance 0.97315
    Number of dimensions: 31, Wasserstein distance 0.97321
    Number of dimensions: 32, Wasserstein distance 0.97324
    Number of dimensions: 33, Wasserstein distance 0.97327
    Number of dimensions: 34, Wasserstein distance 0.97337
    Number of dimensions: 35, Wasserstein distance 0.97345
    Number of dimensions: 36, Wasserstein distance 0.97347
    Number of dimensions: 37, Wasserstein distance 0.97338
    Number of dimensions: 38, Wasserstein distance 0.97339
    Number of dimensions: 39, Wasserstein distance 0.97340


.. code:: ipython3

    print("Selected dimension: {}".format(dim))


.. parsed-literal::

    Selected dimension: 27


PCA and tSNE
------------

Now we perform PCA [^1].

.. code:: ipython3

    zeta = p**-.5 * rs.embed(Y, dim, version='full')

Apply t-SNE

.. code:: ipython3

    from sklearn.manifold import TSNE
    
    tsne_zeta = TSNE(n_components=2, perplexity=30).fit_transform(zeta)

Make dataframes of PCA embedding and t-SNE embedding for plotting

.. code:: ipython3

    zeta_df = pd.DataFrame(zeta[:, :2])
    zeta_df["target"] = np.array(df['target_names'])
    targets = zeta_df["target"].unique()
    targets = sorted(targets)
    labels = df['target']
    
    tsne_zeta_df = pd.DataFrame(tsne_zeta)
    tsne_zeta_df["target"] = np.array(df['target_names'])
    targets = tsne_zeta_df["target"].unique()
    targets = sorted(targets)

Colours dictionary where topics from the same theme have different
shades of the same colour

.. code:: ipython3

    target_colour = {'alt.atheism': 'goldenrod',
                     'comp.graphics': 'steelblue',
                     'comp.os.ms-windows.misc': 'skyblue',
                     'comp.sys.ibm.pc.hardware': 'lightblue',
                     'comp.sys.mac.hardware': 'powderblue',
                     'comp.windows.x': 'deepskyblue',
                     'misc.forsale': 'maroon',
                     'rec.autos': 'limegreen',
                     'rec.motorcycles': 'green',
                     'rec.sport.baseball': 'yellowgreen',
                     'rec.sport.hockey': 'olivedrab',
                     'sci.crypt': 'pink',
                     'sci.electronics': 'plum',
                     'sci.med': 'orchid',
                     'sci.space': 'palevioletred',
                     'soc.religion.christian': 'darkgoldenrod',
                     'talk.politics.guns': 'coral',
                     'talk.politics.mideast': 'tomato',
                     'talk.politics.misc': 'darksalmon',
                     'talk.religion.misc': 'gold'}

Plot PCA on the LHS and PCA + t-SNE on the RHS

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for t in targets:
        t_df = zeta_df[zeta_df['target'] == t]
        ax[0].scatter(t_df[0], t_df[1], marker='o', edgecolor='black',
                      linewidth=0, s=30, label=t, c=target_colour[t])
    ax[0].set_title(f'PCA', fontsize=25)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    
    for t in targets:
        t_df = tsne_zeta_df[tsne_zeta_df['target'] == t]
        ax[1].scatter(t_df[0], t_df[1], marker='o', edgecolor='black',
                      linewidth=0, s=30, label=t, alpha=1, c=target_colour[t])
    ax[1].set_title(f'PCA + t-SNE', fontsize=25)
    ax[1].legend(loc='upper right', bbox_to_anchor=(
        1.51, 1), prop={'size': 15}, markerscale=2)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
    
    # fig.savefig(f"newsgroup.pdf", bbox_inches='tight')



.. image:: text_analysis_files/text_analysis_28_0.png


Hierarchical clustering
-----------------------

Here, we agglomerative clustering with dot products [^2].

First we do this for the centroids of each topic and plot the
dendrogram. Then we do HC on the whole dataset and visualise the output
tree.

.. code:: ipython3

    from sklearn.metrics import pairwise_distances
    from sklearn.cluster import AgglomerativeClustering
    
    def ip_metric(X, Y):
        return np.sum(X * Y)
    
    
    def dp_affinity(X):
        ips = pairwise_distances(X, metric=ip_metric)
        return np.max(ips) - ips

Find the centroids

.. code:: ipython3

    idxs = [np.where(np.array(df['target']) == t)[0]
            for t in sorted(df['target'].unique())]
    t_zeta = np.array([np.mean(zeta[idx, :], axis=0) for idx in idxs])
    t_Y = np.array([np.mean(Y[idx, :],axis = 0) for idx in idxs]).reshape(len(sorted(df['target'].unique())),p)

Topic HC clustering

.. code:: ipython3

    t_dp_clust = AgglomerativeClustering(
        metric=dp_affinity, linkage='average', distance_threshold=0, compute_distances=True, n_clusters=None)
    t_dp_clust = t_dp_clust.fit(t_zeta)

Plot dendrogram

.. code:: ipython3

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    rs.plot_dendrogram(t_dp_clust, orientation='left',
                       labels=sorted(df['target_names'].unique()))
    plt.show()



.. image:: text_analysis_files/text_analysis_37_0.png


Individual document HC clustering

.. code:: ipython3

    dp_clust = AgglomerativeClustering(
        metric=dp_affinity, linkage='average', distance_threshold=0, compute_distances=True, n_clusters=None)
    dp_clust = dp_clust.fit(zeta)

Plot tree

.. code:: ipython3

    colours = [target_colour[label] for label in list(df["target_names"])]

.. code:: ipython3

    rs.plot_HC_clustering(dp_clust, node_colours=colours, internal_node_colour='black', linewidths=.1,
                          edgecolors='black', leaf_node_size=40, fontsize=10, internal_node_size=1, figsize=(10, 7.5
                                                                                                             ))



.. image:: text_analysis_files/text_analysis_42_0.png


References
----------
[^1]: Whiteley, N., Gray, A. and Rubin-Delanchy, P., 2022. Statistical exploration of the Manifold Hypothesis. arXiv preprint arXiv:2208.11665.

[^2]: Gray, A., Modell, A., Rubin-Delanchy, P. and Whiteley, N., 2024. Hierarchical clustering with dot products recovers hidden tree structure. Advances in Neural Information Processing Systems, 36.

