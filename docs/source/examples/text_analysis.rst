Text analysis - 20newsgroup
===========================

.. code:: ipython3

    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups

.. code:: ipython3

    import relastat as rs


.. code:: ipython3

    # replace with own path to data folder:
    path = '/home/ag16115/Documents/phd/codebase_data/'

.. code:: ipython3
    
    newsgroups = fetch_20newsgroups()  # remove=('headers', 'footers', 'quotes')
    # # pprint(list(newsgroups.target_names))
    
    # create dataframe from newsgroup data
    df = pd.DataFrame()
    df["data"] = newsgroups["data"]

.. code:: ipython3

    Y,attributes = rs.matrix_from_text(df, 'data', remove_stopwords=True, clean_text = True, 
                         remove_email_addresses = True, update_stopwords = ['subject'],
                         min_df = 5, max_df = len(df)-1000)

.. code:: ipython3

    for i in range(len(attributes[1])):
        attributes[1][i].update(
            {'label': newsgroups["target_names"][newsgroups["target"][i]]})

.. code:: ipython3

    df1 = pd.DataFrame(
        ['This is a test sentence', 'This is another test sentence', 'This contains an email address: email_address@email.com'], columns=['data'])
    
    Y1, attributes1 = rs.matrix_from_text(df1, 'data', remove_email_addresses=True)