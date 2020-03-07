============
CausalForest
============

The CausalForest class is initiated by setting the respective parameters
which handle the behavior of the forest and trees.
We control the randomness by incrementing seeds whenever randomness is
needed, starting with an initial seed ``seed_counter`` which defaults to 1.

The fitting and prediction process can be parallelized using the
``num_workers`` argument (default is 1), which triggers a parallelization
over processes using `joblib <https://joblib.readthedocs.io/en/latest/>`__.

.. currentmodule:: cforest.forest

.. autoclass:: cforest.forest.CausalForest
   :members:

   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: save
   .. automethod:: load
