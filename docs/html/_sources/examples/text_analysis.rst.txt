Text analysis - 20Newsgroup
===========================

.. code:: ipython3

    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups

.. code:: ipython3

    import relastat as rs

.. code:: ipython3

    # pip install git+ssh://git@github.com/anniegray52/relastat.git 

Import data and create dataframe

.. code:: ipython3

    newsgroups = fetch_20newsgroups() 
    
    df = pd.DataFrame()
    df["data"] = newsgroups["data"]

Create td-idf features

.. code:: ipython3

    import relastat as rs

.. code:: ipython3

    Y, attributes = rs.matrix_from_text(df, 'data', remove_stopwords=True, clean_text=True,
                                        remove_email_addresses=True, update_stopwords=['subject'],
                                        min_df=5, max_df=len(df)-1000)
