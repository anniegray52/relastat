Text analysis - 20Newsgroup
===========================

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups

.. code:: ipython3

    # pip install git+ssh://git@github.com/anniegray52/relastat.git

.. code:: ipython3

    import relastat as rs


Import data and create dataframe

.. code:: ipython3

    newsgroups = fetch_20newsgroups() 
    
    df = pd.DataFrame()
    df["data"] = newsgroups["data"]
    df["target"] = newsgroups["target"]
    df["target_names"] = df.target.apply(
        lambda row: newsgroups["target_names"][row])
    df[['layer1', 'layer2']] = df['target_names'].str.split('.', n=1, expand=True)

Create td-idf features

.. code:: ipython3

    # ## use random sample of data
    n = 5000
    df = df.sample(n=n, replace=False, random_state=22).reset_index(drop=True)

.. code:: ipython3

    Y, attributes = rs.matrix_from_text(df, 'data', remove_stopwords=True, clean_text=True,
                                        remove_email_addresses=True, update_stopwords=['subject'],
                                        min_df=5, max_df=len(df)-1000)

.. code:: ipython3

    (n,p) = Y.shape
    print("n = {}, p = {}".format(n,p))


.. parsed-literal::

    n = 5000, p = 12804


Perform dimension selection using Wasserstein distances

.. code:: ipython3

    rmin = 1
    rmax = 20
    ws = rs.wasserstein_dim_select(Y,rmin = rmin, rmax = rmax)
    dim = rmin + np.argmin(ws)
    print(f'Dimension selected: {np.argmin(ws) + rmin}')
    dim =19


.. parsed-literal::

    100%|██████████| 20/20 [07:43<00:00, 23.18s/it]

.. parsed-literal::

    Dimension selected: 19


.. parsed-literal::

    


PCA and tSNE
------------

Calculate PCA embedding

.. code:: ipython3

    zeta = rs.embed(Y, dim)

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



.. image:: text_analysis_files/text_analysis_22_0.png


Hierarchical clustering
-----------------------

.. code:: ipython3

    from sklearn.metrics import pairwise_distances
    from sklearn.cluster import AgglomerativeClustering
    
    def ip_metric(X, Y):
        return np.sum(X * Y)
    
    
    def ip_affinity(X):
        ips = pairwise_distances(X, metric=ip_metric)
        return np.max(ips) - ips

Find the centroids

.. code:: ipython3

    idxs = [np.where(np.array(df['target']) == t)[0]
            for t in sorted(df['target'].unique())]
    t_zeta = np.array([np.mean(zeta[idx, :], axis=0) for idx in idxs])
    t_Y = np.array([np.mean(Y[idx, :],axis = 0) for idx in idxs]).reshape(len(sorted(df['target'].unique())),p)

Perform hierarchical clustering with dot products from:
https://arxiv.org/abs/2305.15022

.. code:: ipython3

    ip_t_clust = AgglomerativeClustering(
        metric=ip_affinity, linkage='average', distance_threshold=0, n_clusters=None)
    ip_t_clust = ip_t_clust.fit(t_zeta)

Function to plot dendrogram

.. code:: ipython3

    from scipy.cluster.hierarchy import dendrogram
    
    
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram
    
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
    
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
    
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

.. code:: ipython3

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(ip_t_clust, orientation = 'left', labels=sorted(df['target_names'].unique()))
    plt.show()



.. image:: text_analysis_files/text_analysis_31_0.png

