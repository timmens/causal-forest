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
from algorithms.tree.causaltree import predict_causaltree


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
        cforest, keys=range(num_trees), names=["tree_id", "node_id"]
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


def predict_causalforest(cforest, x):
    """Predicts individual treatment effects for a causal forest.

    Predicts individual treatment effects for new observed features *x*
    on a fitted causal forest *cforest*.

    Args:
        cforest (pd.DataFrame): fitted causal forest represented in a multi-
            index pd.DataFrame consisting of several fitted causal trees
        x (np.array): 2d array of new observations for which we predict the
            individual treatment effect.

    Returns:
        predictions (np.array): 1d array of treatment predictions.

    """
    num_trees = len(cforest.groupby(level=0))
    n, p = x.shape

    predictions = np.empty((num_trees, n))
    for i in range(num_trees):
        predictions[i, :] = predict_causaltree(cforest.loc[i], x)

    predictions = predictions.mean(axis=0)
    return predictions
