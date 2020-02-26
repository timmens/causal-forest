"""
Module to fit a causal tree.

This module provides functions to fit a causal tree and predict treatment
effects using a fitted causal tree.
"""
from itertools import count

import numpy as np
import pandas as pd
from numba import njit


def fit_causaltree(X, t, y, critparams=None):
    """Fit a causal tree on given data.

    Wrapper function for `_fit_node`. Sets default parameters for
    *crit_params* and calls internal fitting function `_fit_node`. Returns
    fitted tree as a pd.DataFrame.

    Args:
        X (np.array): Data on features
        t (np.array): Data on treatment status
        y (np.array): Data on outcomes
        critparams (dict): Dictionary containing information on when to stop
            splitting further, i.e., minimum number of leafs and maximum
            depth of the tree. Default is set to 'min_leaf' = 4 and
            'max_depth' = 20.

    Returns:
        ctree (pd.DataFrame): the fitted causal tree represented as a pandas
            data frame. If honest is true ctree is an honest causal tree and
            a regular otherwise.

    """
    if critparams is None:
        critparams = {
            "min_leaf": 4,
            "max_depth": 20,
        }

    # initialize counter object and id_params
    counter = count(0)
    rootid = next(counter)
    idparams = {"counter": counter, "id": rootid, "level": 0}

    # initialize index (the root node considers all observations).
    n = len(y)
    index = np.full((n,), True)

    # fit tree
    ctree_array = _fit_node(
        X=X, t=t, y=y, index=index, critparams=critparams, idparams=idparams,
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
    columns_to_int = column_names[:5]

    ct = pd.DataFrame(ctree_array, columns=column_names)
    ct[columns_to_int] = ct[columns_to_int].astype("Int64")
    ct = ct.set_index("id").sort_index()

    return ct


def _fit_node(X, t, y, index, critparams, idparams):
    """Fits a single decision tree node recursively.

    Recursively split feature space until stopping criteria in *crit_params*
    are reached. In each level of recursion fit a single node.

    Args:
        X (np.array): data on features
        t (np.array): data on treatment status
        y (np.array): data on outcomes
        index (np.array): boolean index indicating which observations (rows)
            of the data to consider for split.
        critparams (dict): dictionary containing information on when to stop
            splitting further, i.e., minimum number of leafs and maximum
            depth of the tree.
        idparams (dict): dictionary containing identification information of
            a single node. That is, a unique id number, the level in the tree,
            and a counter object which is passed to potential children of node.

    Returns:
        out (np.array): array containing information on the splits, with
            columns representing (nodeid, left_childid, right_childid, level,
            split_feat).

    """
    level = idparams["level"]
    nodeid = idparams["id"]

    tmp = _find_optimal_split(
        X=X, t=t, y=y, index=index, min_leaf=critparams["min_leaf"],
    )

    if tmp is None or level == critparams["max_depth"]:
        # if we do not split the node must be a leaf, hence we add the
        # treatment effect to the output.
        treat_effect = _estimate_treatment_effect(y[index], t[index])

        info = np.array(
            [nodeid, np.nan, np.nan, level, np.nan, np.nan, treat_effect,]
        ).reshape((1, 7))
        return info
    else:
        left, right, split_feat, split_value = tmp

        leftid = next(idparams["counter"])
        rightid = next(idparams["counter"])

        info = np.array([nodeid, leftid, rightid, level])

        split_info = np.array([split_feat, split_value, np.nan])
        info = np.append(info, split_info).reshape((1, 7))

        idparams_left = idparams.copy()
        idparams_left["id"] = leftid
        idparams_left["level"] += 1

        idparams_right = idparams.copy()
        idparams_right["id"] = rightid
        idparams_right["level"] += 1

        out_left = _fit_node(
            X=X,
            t=t,
            y=y,
            index=left,
            critparams=critparams,
            idparams=idparams_left,
        )
        out_right = _fit_node(
            X=X,
            t=t,
            y=y,
            index=right,
            critparams=critparams,
            idparams=idparams_right,
        )

        out = np.vstack((info, out_left))
        out = np.vstack((out, out_right))
        return out


def _find_optimal_split(X, t, y, index, min_leaf):
    """Compute optimal split and splitting information.

    For given data, for each feature, go through all valid splitting points and
    find the split value and split variable which result in the lowest overall
    loss.

    Args:
        X (np.array): data on features
        t (np.array): data on treatment status
        y (np.array): data on outcomes
        index (np.array): boolean index indicating which observations (rows)
            of the data to consider for split.
        min_leaf (int): Minimum number of observations of each type (treated,
            untreated) allowed in a leaf; has to be greater than 1.

    Returns:
        left (np.array): boolean index representing the observations falling
            in the left leaf.
        right (np.array): boolean index representing the observations falling
            in the right leaf.
        split_feat (int): index of feature on which the optimal split occurs.
        split_value (float): value of feature on which the optimal split
            occurs.

    """
    _, p = X.shape
    split_feat = None
    split_value = None
    split_index = None
    loss = np.inf

    for j in range(p):
        # loop through features

        index_sorted = np.argsort(X[index, j])
        yy = y[index][index_sorted]
        xx = X[index, j][index_sorted]
        tt = t[index][index_sorted]

        yy_transformed = _transform_outcome(y=yy, t=tt)

        splitting_indices = _compute_valid_splitting_indices(
            t=tt, min_leaf=min_leaf
        )

        # loop through observations
        tmp = _find_optimal_split_inner_loop(
            splitting_indices=splitting_indices,
            x=xx,
            t=tt,
            y=yy,
            y_transformed=yy_transformed,
            min_leaf=min_leaf,
        )
        jloss, jsplit_value, jsplit_index = tmp

        if jloss < loss and jsplit_index is not None:
            split_feat = j
            split_value = jsplit_value
            split_index = jsplit_index
            loss = jloss

    # check if any split has occured.
    if loss == np.inf:
        return None

    # create index of observations falling in left and right leaf, respectively
    index_sorted = np.argsort(X[index, split_feat])
    left, right = _retrieve_index(
        index=index, sorted_subset_index=index_sorted, split_index=split_index
    )
    return left, right, split_feat, split_value


def _find_optimal_split_inner_loop(
    splitting_indices, x, t, y, y_transformed, min_leaf
):
    """Find the optimal splitting value for data on a single feature.

    Finds the optimal splitting value (in the data) given data on a single
    feature. Note that the algorithm is essentially not different from naively
    searching for a minimum; however, since this is computationally very costly
    this function implements a dynamic updating procedure, in which the sums
    get updated during the loop. See "Elements of Statistical Learning" for a
    reference on the classical algorithm.

    Args:
        splitting_indices (np.array): valid splitting indices.
        x (np.array): data on a single feature.
        t (np.array): data on treatment status.
        y (np.array): data on outcomes.
        y_transformed (np.array): data on transformed outcomes.

    Returns:
         - (np.inf, None, None): if *splitting_indices* is empty
         - (minimal_loss, split_value, split_index): if *splitting_indices* is
            not empty, where minimal_loss denotes the loss occured when
            splitting the feature axis at the split_value (= x[split_index]).

    """
    if len(splitting_indices) == 0:
        return np.inf, None, None

    # initialize number of observations
    i0 = splitting_indices[0]
    n_1l = int(np.sum(t[: (i0 + 1)]))
    n_0l = int(np.sum(~t[: (i0 + 1)]))
    n_1r = int(np.sum(t[(i0 + 1) :]))
    n_0r = len(t) - n_1l - n_0l - n_1r

    # initialize dynamic sums
    sum_1l = y[t][: (i0 + 1)].sum()
    sum_0l = y[~t][: (i0 + 1)].sum()
    sum_1r = y[t][(i0 + 1) :].sum()
    sum_0r = y[~t][(i0 + 1) :].sum()

    split_value = x[i0]
    split_index = i0
    minimal_loss = _compute_global_loss(
        sum_0l=sum_0l,
        sum_1l=sum_1l,
        sum_0r=sum_0r,
        sum_1r=sum_1r,
        n_0l=n_0l,
        n_1l=n_1l,
        n_0r=n_0r,
        n_1r=n_1r,
        y_transformed=y_transformed,
        i=i0,
    )

    for i in splitting_indices[1:]:
        if t[i]:
            sum_1l += y[i]
            sum_1r -= y[i]
            n_1l += 1
            n_1r -= 1
        else:
            sum_0l += y[i]
            sum_0r -= y[i]
            n_0l += 1
            n_0r -= 1

        # this should not happen but it does for some reason (Bug alarm!)
        if n_0r < min_leaf or n_1r < min_leaf:
            break
        if n_0l < min_leaf or n_1l < min_leaf:
            continue

        global_loss = _compute_global_loss(
            sum_0l=sum_0l,
            sum_1l=sum_1l,
            sum_0r=sum_0r,
            sum_1r=sum_1r,
            n_0l=n_0l,
            n_1l=n_1l,
            n_0r=n_0r,
            n_1r=n_1r,
            y_transformed=y_transformed,
            i=i,
        )
        if global_loss < minimal_loss:
            split_value = x[i]
            split_index = i
            minimal_loss = global_loss

    return minimal_loss, split_value, split_index


def _compute_global_loss(
    sum_1l, n_1l, sum_0l, n_0l, sum_1r, n_1r, sum_0r, n_0r, y_transformed, i
):
    """Compute global loss when splitting at index *i*.

    Computes global loss when splitting the observation set at index *i*
    using the dynamically updated sums and number of observations.

    Args:
        sum_1l (float): Sum of outcomes of treated observations left to the
            potential split at index *i*.
        n_1l (int): Number of treated observations left to the potential split
            at index *i*.
        sum_0l (float): Sum of outcomes of untreated observations left to the
            potential split at index *i*.
        n_0l (int): Number of untreated observations left to the potential
            split at index *i*.
        sum_1r (float): Sum of outcomes of treated observations right to the
            potential split at index *i*.
        n_1r (int): Number of treated observations right to the potential split
            at index *i*.
        sum_0r (float): Sum of outcomes of untreated observations right to the
            potential split at index *i*.
        n_0r (int): Number of untreated observations right to the potential
            split at index *i*.
        y_transformed (np.array): Transformed outcomes.
        i (int): Index at which to split.

    Returns:
        global_loss (float): The loss when splitting at index *i*.

    """
    left_te = _compute_treatment_effect_raw(sum_1l, n_1l, sum_0l, n_0l)
    right_te = _compute_treatment_effect_raw(sum_1r, n_1r, sum_0r, n_0r)

    left_loss = ((y_transformed[: (i + 1)] - left_te) ** 2).sum()
    right_loss = ((y_transformed[(i + 1) :] - right_te) ** 2).sum()

    global_loss = left_loss + right_loss
    return global_loss


def _compute_valid_splitting_indices(t, min_leaf):
    """Compute valid split indices for treatment array *t* given *min_leaf*.

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
    """Transform outcome.

    Transforms outcome using approximate propensity scores. Equation is as
    follows: y_transformed_i = 2 * y_i * t_i - 2 * y_i * (1 - t_i), where t_i
    denotes the treatment status of the ith individual. This object is
    equivalent to the individual treatment effect in expectation.

    Args:
        y (np.array): data on outcomes.
        t (np.arra): boolean data on treatment status.

    Returns:
        y_transformed (np.array): the transformed outcome.

    Example:
    >>> import numpy as np
    >>> y = np.array([-1, 0, 1])
    >>> t = np.array([True, True, False])
    >>> _transform_outcome(y, t)
    array([-2,  0, -2])

    """
    y_transformed = 2 * y * t - 2 * y * (1 - t)
    return y_transformed


def _estimate_treatment_effect(y, t):
    """Estimate the average treatment effect.

    Estimates average treatment effect (ATE) using outcomes *y* and treatment
    status *t*.

    Args:
        y (np.array): data on outcomes.
        t (np.array): boolean data on treatment status.

    Returns:
        out (float): the estimated treatment effect.

    Example:
    >>> import numpy as np
    >>> y = np.array([-1, 0, 1, 2, 3, 4])
    >>> t = np.array([False, False, False, True, True, True])
    >>> _estimate_treatment_effect(y, t)
    3.0

    """
    out = y[t].mean() - y[~t].mean()
    return out


def _retrieve_index(index, sorted_subset_index, split_index):
    """Get index of left and right leaf relative to complete data set.

    Given an array of indices *index* of length of the original data set, and
    a sorted index array *index_sorted* (sorted with respect to the feature
    on which we split; see function _find_optimal_split) and an index on which
    we want to split (*split_index*), `_retrieve_index` computes two indices
    (left and right) the same length as *index* corresponding to observations
    falling falling left and right to the splitting point, respectively.

    Args:
        index (np.array): boolean array indicating which observations (rows)
            of the data to consider for split.
        sorted_subset_index (np.array): array containing indices, sorted with
            respect to the feature under consideration. Length is equal to the
            number of True values in *index*.
        split_index (int): index in *sorted_subset_index* corresponding to the
            split.

    Returns:
        out: 2d tuple containing np.arrays left_index and right_index the same
            length as *index*

    Example:
    >>> import numpy as np
    >>> from pprint import PrettyPrinter
    >>> index = np.array([True, True, True, False, False, True])
    >>> sorted_subset_index = np.array([0, 3, 1, 2])
    >>> split_index = 1
    >>> PrettyPrinter().pprint(_retrieve_index(index, sorted_subset_index,
    ... split_index))
    (array([ True, False, False, False, False,  True]),
     array([False,  True,  True, False, False, False]))

    """
    # Not solving the bug:
    if split_index is None:
        return index

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

    out = left_index, right_index
    return out


@njit
def _compute_treatment_effect_raw(
    sum_treated, n_treated, sum_untreated, n_untreated
):
    """Compute the average treatment effect.

    Computes the average treatment effect (ATE) using the sum of outcomes of
    treated and untreated observations (*sum_treated* and *sum_untreated*) and
    the number of treated and untreated observations (*n_treated* and
    *n_untreated*).

    Args:
        sum_treated (float): sum of outcomes of treatment individuals.
        n_treated (int): number of treated individuals.
        sum_untreated (float): sum of outcomes of untreated individuals.
        n_untreated (int): number of untreated individuals.

    Returns:
        out (float): the estimated treatment effect

    Example:
    >>> sum_t, n_t = 100, 10.0
    >>> sum_unt, n_unt = 1000, 20.0
    >>> _compute_treatment_effect_raw(sum_t, n_t, sum_unt, n_unt)
    -40.0

    """
    out = sum_treated / n_treated - sum_untreated / n_untreated
    return out


def predict_causaltree(ctree, X):
    """Predicts individual treatment effects for a causal tree.

    Predicts individual treatment effects for new observed features *x*
    on a fitted causal tree *ctree*.

    Args:
        ctree (pd.DataFrame): fitted causal tree represented in a pd.DataFrame
        X (np.array): data on new observations

    Returns:
        predictions (np.array): treatment predictions.

    """
    n = len(X)
    predictions = np.empty((n,))
    for i, row in enumerate(X):
        predictions[i] = _predict_row_causaltree(ctree, row)

    return predictions


def _predict_row_causaltree(ctree, row):
    """Predicts treatment effect for a single individual.

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
