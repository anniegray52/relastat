.. RelaStat documentation master file, created by
   sphinx-quickstart on Thu Dec  7 09:58:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RelaStat's documentation!
====================================

This has the documentation for RelaStat, a Python package for statistical
analysis of relational data.  The way it is written in here might be a bit confusing because it isn't actually a package yet. 

To use the functions, all the files in the `relastat` folder need to be in the same directory as the file you are working in.  Then, you can import the functions like this:
This is shown in the `tutorial.ipynb` file. In this documentation it for a function written as: `data_preparation.graph_functions.matrix_from_tables`` you would import 
all the functions in that file by writing: `from relastat.data_preparation.graph_functions import *`.  Then you can use the functions by writing `matrix_from_tables()`.

Hopefully this will be a package soon, but for now this is how it works.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
