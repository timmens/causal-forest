========
Overview
========

CForest provides classes and functions to estimate heterogeneous treatment
effects in a potential outcome framework.

The algorithms which are implemented in CForest draw heavily on the ideas
formulated in Athey and Imbens (2016) and Athey and Wager (2019), who
first proposed the Causal Tree and Causal Forest algorithms.

Here it is appropriate to also refer to Athey, Tibshirani and Wager (2019)
who combine and generalize the ideas of causal and random forests.
Further,
they provide an `R` package (https://github.com/grf-labs/grf) which can be
used to compute everything CForest computes and much more.

Example
=======

A complete working example can be found in section ``example``.


Warnings
========

Originally there were two reasons for the creation of CForest, given that
there already exists a well maintained package.
First, through the implementation of algorithms one often learns more
about the inner workings of given algorithms.
And second, the implementation in `grf` is written in C++ and wrapped in
`R`, which makes it very hard to explain details of the implementation
to students and researchers not trained in C++.
This is why we were interested in a `Python` implementation that uses
`numpy` and `numba`, which allow for great readibility.

As of right now CForest is still under development and should not be used
other than for experimental reasons.
An official version will become more likely once we benchmarked our
implementation to `grf` with positive results.


References
==========

- Athey and Imbens, 2016, `Estimation and Inference of Heterogeneous Treatment Effects using Random Forests <https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839>`__

- Athey and Wager, 2019, `Recursive partitioning for heterogeneous causal effects <https://www.pnas.org/content/113/27/7353>`__

- Athey, Tibshirani and Wager, 2019 `Generalized random forests <https://projecteuclid.org/euclid.aos/1547197251>`__
