============
CausalForest
============

The CausalForest class can be initiated with parameters ``forestparams``,
which denote the parameters that handle the behavior of the forest,
``treeparams``, which denote the parameters that handle the behavior of the
individual trees and ``seed_counter``, which denotes the number on which the
seed generator starts producing seed sequences.

If no values are passed the default values get used.

.. code-block:: python

   forestparams = {"num_trees": 100, "ratio_features_at_split": 0.7}

   treeparams = {"min_leaf": 4, "max_depth": 25}

   seed_counter = 1

.. currentmodule:: cforest.forest

.. autoclass:: cforest.forest.CausalForest
      :noindex:
