#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Module to fit a causal tree.
"""
from itertools import count

import numpy as np
import pandas as pd
from numba import njit


def fit_causaltree(y, t, x, crit_params=None, func_params=None):
    """ Wrapper function for `_fit_node`. Sets default parameters for
    *crit_params* and *func_params* and calls internal fitting function
    `_fit_node`. Returns fitted tree as a pd.DataFrame.

    Args:
        y:
        t:
        x:
        crit_params:
        func_params:

    Returns:
        ctree (pd.DataFrame):
    """

    if crit_params is None:
        crit_params = {
            "min_leaf": 3,
            "max_depth": 25,
        }

    if func_params is None:

        def metric(outcomes, estimate):
            return np.sum((outcomes - estimate) ** 2)

        def weight_loss(left_loss, right_loss, w1, w2):
            return left_loss + right_loss

        func_params = {
            "metric": metric,
            "weight_loss": weight_loss,
        }

    # initialize counter object and id_params
    counter = count()
    rootid = next(counter)
    id_params = {"counter": counter, "id": rootid, "level": 0}

    # initialize index
    n = len(y)
    index = np.full((n,), True)

    # fit tree
    ctree_array = _fit_node(
        y=y,
        t=t,
        x=x,
        index=index,
        crit_params=crit_params,
        func_params=func_params,
        id_params=id_params,
    )

    column_names = [
        "id",
        "left_child",
        "right_child",
        "level",
        "split_feat",
        "split_value",
        "treat_effect",
    ]

    ctree = pd.DataFrame(ctree_array, columns=column_names)
    ctree[["id", "left_child", "right_child", "level", "split_feat"]] = ctree[
        ["id", "left_child", "right_child", "level", "split_feat"]
    ].astype("Int64")

    ctree = ctree.set_index("id").sort_index()
    return ctree


def _fit_node(y, t, x, index, crit_params, func_params, id_params):
    """
    Recursively split feature space until stopping criteria in *crit_params*
    are reached.

    Args:
        y:
        t:
        x:
        index:
        crit_params:
        func_params:
        id_params:

    Returns:

    """
    level = id_params["level"]
    nodeid = id_params["id"]

    # column_names = [
    #     "id",
    #     "left_child",
    #     "right_child",
    #     "level",
    #     "split_feat",
    #     "split_value",
    #     "treat_effect",
    # ]

    # df_out = pd.DataFrame(columns=column_names)

    tmp = _find_optimal_split(y, t, x, index, crit_params["min_leaf"],)

    if tmp is None or level == crit_params["max_depth"]:
        # if we do not split the node must be a leaf, hence we add the
        # treatment effect
        treat_effect = _estimate_treatment_effect(y[index], t[index])

        info = np.array(
            [nodeid, np.nan, np.nan, level, np.nan, np.nan, treat_effect,]
        ).reshape((1, 7))

        # to_append = pd.Series(info, column_names)
        # return df_out.append(to_append, ignore_index=True)
        return info
    else:
        left, right, split_feat, split_value = tmp

        leftid = next(id_params["counter"])
        rightid = next(id_params["counter"])

        info = np.array([nodeid, leftid, rightid, level])

        split_info = np.array([split_feat, split_value, np.nan])
        info = np.append(info, split_info).reshape((1, 7))
        # to_append = pd.Series(info, column_names)
        # df_out = df_out.append(to_append, ignore_index=True)

        id_params_left = id_params.copy()
        id_params_left["id"] = leftid
        id_params_left["level"] += 1

        id_params_right = id_params.copy()
        id_params_right["id"] = rightid
        id_params_right["level"] += 1

        out_left = _fit_node(
            y=y,
            t=t,
            x=x,
            index=left,
            crit_params=crit_params,
            func_params=func_params,
            id_params=id_params_left,
        )
        out_right = _fit_node(
            y=y,
            t=t,
            x=x,
            index=right,
            crit_params=crit_params,
            func_params=func_params,
            id_params=id_params_right,
        )

        # df_out = df_out.append(out_left, ignore_index=True)
        # df_out = df_out.append(out_right, ignore_index=True)
        out = np.vstack((info, out_left))
        out = np.vstack((out, out_right))

        # return df_out
        return out


def _find_optimal_split(y, t, x, index, min_leaf):
    """

    Args:
        y:
        t:
        x:
        index:
        min_leaf:

    Returns:

    """
    _, p = x.shape
    split_feat = None
    split_value = None
    split_index = None
    loss = np.inf

    for j in range(p):
        # loop through features

        index_sorted = np.argsort(x[index, j])
        yy = y[index][index_sorted]
        xx = x[index, j][index_sorted]
        tt = t[index][index_sorted]

        yy_transformed = _transform_outcome(yy, tt)

        splitting_indices = _compute_valid_splitting_indices(tt, min_leaf)

        # loop through observations
        tmp = _find_optimal_split_observation_loop(
            splitting_indices, yy, yy_transformed, xx, tt, loss
        )
        jloss, jsplit_value, jsplit_index = tmp

        if jloss < loss:
            split_feat = j
            split_value = jsplit_value
            split_index = jsplit_index
            loss = jloss

    # check if any split has occured.
    if loss == np.inf:
        return None

    # create index of observations falling in left and right leaf, respectively
    index_sorted = np.argsort(x[index, split_feat])
    left, right = _retrieve_index(index, index_sorted, split_index)

    return left, right, split_feat, split_value


def _compute_valid_splitting_indices(t, min_leaf):
    """
    Given an array *t* of treatment status and an integer *min_leaf* --denoting
    the minimum number of allowed observations of each type in a leaf node--
    computes a sequence of indices on which we can split *t* and get that each
    resulting side contains a minimum of *min_leaf* treated and untreated
    observations. Returns an empty sequence if no split is possible.

    Args:
        t (np.array): 1d array containing the treatment status as treated =
            True and untreated = False.
        min_leaf (int): Minimum number of observations of each type (treated,
            untreated) allowed in a leaf; has to be greater than 1.

    Returns:
        out (np.array): a sequence of indices representing valid splitting
            points.

    """
    out = np.arange(0)

    n = len(t)
    if n < 2 * min_leaf:
        return out

    # find first index at which *min_leaf* treated obs. are in left split
    left_index_treated = np.argmax(np.cumsum(t) == min_leaf)
    if left_index_treated == 0:
        return out

    # find first index at which *min_leaf* untreated obs. are in left split
    left_index_untreated = np.argmax(np.cumsum(~t) == min_leaf)
    if left_index_untreated == 0:
        return out

    # first split at which both treated and untreated occure more often than
    # *min_leaf* is given by the maximum.
    left = np.max([left_index_treated, left_index_untreated])

    # do the same for right side
    right_index_treated = np.argmax(np.cumsum(np.flip(t)) == min_leaf)
    if right_index_treated == 0:
        return out

    right_index_untreated = np.argmax(np.cumsum(np.flip(~t)) == min_leaf)
    if right_index_untreated == 0:
        return out

    right = n - np.max([right_index_treated, right_index_untreated])

    if left > right - 1:
        return out
    else:
        out = np.arange(left, right - 1)
        return out


def _transform_outcome(y, t):
    """
    Transforms outcome using naive propensity scores (Prob[`t`=1] = 1/2).
    TODO: Implement general transformation using propensity scores.

    :param y: (n,) np.array containing outcomes
    :param t: (n,) np.array (bool) containing treatment status
    :return: (n,) np.array of transformed outcomes
    """
    y_transformed = 2 * y * t - 2 * y * (1 - t)

    return y_transformed


def _find_optimal_split_observation_loop(
    splitting_indices, yy, yy_transformed, xx, tt, loss
):
    """

    Args:
        splitting_indices:
        yy:
        yy_transformed:
        xx:
        tt:
        loss:

    Returns:

    """
    if len(splitting_indices) == 0:
        return loss, None, None

    split_value = None
    split_index = None
    squared_sum_transformed = (yy_transformed ** 2).sum()
    minimal_loss = loss - squared_sum_transformed

    i0 = splitting_indices[0]
    n_1l = np.sum(tt[: (i0 + 1)])
    n_0l = np.sum(~tt[: (i0 + 1)])
    n_1r = np.sum(tt[(i0 + 1) :])
    n_0r = len(tt) - n_1l - n_0l - n_1r

    sum_1l = yy[tt][: (i0 + 1)].sum()
    sum_0l = yy[~tt][: (i0 + 1)].sum()
    sum_1r = yy[tt][(i0 + 1) :].sum()
    sum_0r = yy[~tt][(i0 + 1) :].sum()

    left_te = _compute_treatment_effect_raw(sum_1l, n_1l, sum_0l, n_0l)
    right_te = _compute_treatment_effect_raw(sum_1r, n_1r, sum_0r, n_0r)

    left_loss = _compute_loss_raw_left(yy_transformed, i0, left_te)
    right_loss = _compute_loss_raw_right(yy_transformed, i0, right_te)

    global_loss = left_loss + right_loss
    if global_loss < minimal_loss:
        split_value = xx[i0]
        split_index = i0
        minimal_loss = global_loss

    for i in splitting_indices[1:]:

        if tt[i]:
            sum_1l += yy[i]
            sum_1r -= yy[i]
            n_1l += 1
            n_1r -= 1
        else:
            sum_0l += yy[i]
            sum_0r -= yy[i]
            n_0l += 1
            n_0r -= 1

        left_te = _compute_treatment_effect_raw(sum_1l, n_1l, sum_0l, n_0l)
        right_te = _compute_treatment_effect_raw(sum_1r, n_1r, sum_0r, n_0r)

        left_loss = _compute_loss_raw_left(yy_transformed, i, left_te)
        right_loss = _compute_loss_raw_right(yy_transformed, i, right_te)

        global_loss = left_loss + right_loss
        # global_loss = loss_weighting(
        #     left_loss, right_loss, i + 1, len(yy) - i - 1
        # )
        if global_loss < minimal_loss:
            split_value = xx[i]
            split_index = i
            minimal_loss = global_loss

    return minimal_loss + squared_sum_transformed, split_value, split_index


def _estimate_treatment_effect(y, t):
    """
    Estimates average treatment effect (ATE) using outcomes *y* and treatment
    status *t*.

    Args:
        y (np.array): 1d array containing outcomes
        t (np.array): 1d array containing the treatment status as treated =
            True and untreated = False.

    Returns:
        out (float): the estimated treatment effect

    """
    out = y[t].mean() - y[~t].mean()
    return out


@njit
def _weight_loss(left_loss, right_loss, n_left, n_right):
    """
    Given loss in a left leaf (`left_loss`)  and right leaf (`right_loss`) and
    the number of observations falling in the left and right leaf, `n_left` and
    `n_right`, respectively, computes a weighted combination of the losses
    and returns a single scalar output.

    :param left_loss: loss in left leaf
    :param right_loss: loss in right leaf
    :param n_left: no. of observations falling in left leaf
    :param n_right: no. of observations falling in right leaf
    :return: weightes loss scalar (float)
    """
    left = (n_left / (n_left + n_right)) * left_loss
    right = (n_right / (n_left + n_right)) * right_loss

    return left + right


def _retrieve_index(index, sorted_subset_index, split_index):
    """
    Given an array of indices *index* of length of the original data set, and
    a sorted index array *index_sorted* (sorted with respect to the feature
    on which we split; see function _find_optimal_split) and an index on which
    we want to split (*split_index*), `_retrieve_index` computes two indices
    (left and right) the same length as *index* corresponding to observations
    falling falling left and right to the splitting point, respectively.

    Args:
        index (np.array): boolean
        sorted_subset_index (np.array): int
        split_index (int):

    Returns:
        out: 2d tuple containing np.arrays left_index and right_index the same
            length as *index*

    """
    left = sorted_subset_index[: (split_index + 1)]
    right = sorted_subset_index[(split_index + 1) :]
    nonzero_index = np.nonzero(index)[0]

    # initialize new indices
    n = len(index)
    left_index = np.full((n,), False)
    right_index = np.full((n,), False)

    # fill nonzero values
    left_index[nonzero_index[left]] = True
    right_index[nonzero_index[right]] = True

    # global_split_index = nonzero_index[index_sorted[split_index]]

    out = left_index, right_index
    return out


@njit
def _compute_treatment_effect_raw(
    sum_treated, n_treated, sum_untreated, n_untreated
):
    """
    Computes average treatment effect (ATE) using the sum of outcomes of
    treated and untreated observations (*sum_treated* and *sum_untreated*) and
    the number of treated and untreated observations (*n_treated* and
    *n_untreated*).

    Args:
        sum_treated:
        n_treated:
        sum_untreated:
        n_untreated:

    Returns:
        out (float): the estimated treatment effect

    """
    out = sum_treated / n_treated - sum_untreated / n_untreated
    return out


def _compute_loss_raw_left(yy_transformed, i, te):
    """

    Args:
        yy_transformed:
        i:
        te:

    Returns:

    """
    return te ** 2 - 2 * te * yy_transformed[: (i + 1)].sum()


def _compute_loss_raw_right(yy_transformed, i, te):
    """

    Args:
        yy_transformed:
        i:
        te:

    Returns:

    """
    return te ** 2 - 2 * te * yy_transformed[(i + 1) :].sum()


def predict_causaltree(ctree, x):
    """
    Predicts individual treatment effects for new observed features *x*
    on a fitted causal tree *ctree*.

    Args:
        ctree (pd.DataFrame): fitted causal tree represented in a pd.DataFrame
        x (np.array): 2d array of new observations for which we predict the
            individual treatment effect.

    Returns:
        predictions (np.array): 1d array of treatment predictions.

    """
    n, _ = x.shape
    predictions = np.empty((n,))
    for i, row in enumerate(x):
        predictions[i] = _predict_row_causaltree(ctree, row)

    return predictions


def _predict_row_causaltree(ctree, row):
    """
    Predicts individual treatment effects for new observed features *row* for a
    single individual on a fitted causal tree *ctree*.

    Args:
        ctree (pd.DataFrame): fitted causal tree represented in a pd.DataFrame
        row (np.array): 1d array of features for single new observation

    Returns:
        prediction (float): treatment prediction.

    """
    current_id = 0
    while np.isnan(ctree.loc[current_id, "treat_effect"]):
        split_feat = ctree.loc[current_id, "split_feat"]
        go_left = row[split_feat] <= ctree.loc[current_id, "split_value"]

        if go_left:
            current_id = ctree.loc[current_id, "left_child"]
        else:
            current_id = ctree.loc[current_id, "right_child"]

    return ctree.loc[current_id, "treat_effect"]
