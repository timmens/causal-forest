#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Module to fit a causal forest.

This module uses the causal tree algorithm from `algorithm.causaltree.py` to
fit a causal forest on given data.
"""
from itertools import count

import numpy as np
import pandas as pd

from algorithms.tree.causaltree import fit_causaltree


def fit_causalforest(
    y, t, x, num_trees, crit_params_ctree=None, seed_counter=1000
):
    """Fits a causal forest on given data.

    Fits a causal forest using data on outcomes *y*, treatment status *t*
    and features *x*. Forest will be made up of *num_trees* causal trees with
    parameters *crit_params_ctree*.

    Args:
        y:
        t:
        x:
        num_trees:
        crit_params_ctree:
        seed_counter:

    Returns:
        cforest (pd.DataFrame):

    """
    n = len(y)
    counter = count(seed_counter)

    cforest = []
    for _i in range(num_trees):
        seed = next(counter)
        resample_index = _draw_resample_index(n, seed)
        ctree = fit_causaltree(
            y[resample_index],
            t[resample_index],
            x[resample_index],
            crit_params_ctree,
        )
        cforest.append(ctree)

    cforest = pd.concat(
        cforest, keys=range(1, num_trees + 1), names=["tree_id", "node_id"]
    )

    return cforest


def _draw_resample_index(n, seed):
    """Compute vector of randomly drawn indices with replacement.

    Draw indices with replacement from the discrete uniform distribution
    on {0,...,n-1}. We control the randomness by setting the seed to *seed*.
    If *seed* = -1 we return all indices {0,1,2,...,n-1} for debugging.

    Args:
        n (int): Upper bound for indices and number of indices to draw
        seed (int): Random number seed.

    Returns:
        indices (np.array): Resample indices (resampled with replacement)

    """
    if seed == -1:
        return np.arange(n)
    np.random.seed(seed)
    indices = np.random.randint(0, n, n)
    return indices
