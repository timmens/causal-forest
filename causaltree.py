# Module to fit a causal tree on numpy arrays using numba
# TODO: see issues on github
from itertools import count

import numpy as np
import pandas as pd
from numba import njit


def _compute_valid_splitting_indices(t, min_leaf):
    """
    Computes potential splitting point indices.

    :param t: (n, ) np.array (bool) treatment status sorted wrt to features
    :param min_leaf:
    :return: sequence of indices to be considered for a split
    """
    nn = len(t)
    if nn <= min_leaf:
        return np.arange(0)

    tmp = np.where(np.cumsum(t) == min_leaf)[0]
    if tmp.size == 0:
        return np.arange(0)
    else:
        left_treated = tmp[0]
    tmp = np.where(np.cumsum(~t) == min_leaf)[0]
    if tmp.size == 0:
        return np.arange(0)
    else:
        left_untreated = tmp[0]
    left = max(left_treated, left_untreated)

    tmp = np.where(np.cumsum(np.flip(t)) == min_leaf)[0]
    if tmp.size == 0:
        return np.arange(0)
    else:
        right_treated = tmp[0]
    tmp = np.where(np.cumsum(np.flip(~t)) == min_leaf)[0]
    if tmp.size == 0:
        return np.arange(0)
    else:
        right_untreated = tmp[0]
    right = nn - 1 - max(right_treated, right_untreated)

    if left > right - 1:
        return np.arange(0)
    else:
        return np.arange(left, right - 1)


@njit
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


@njit
def _estimate_treatment_effect(y, t):
    """
    Estimates average treatment effect (ATE) using outcomes `y` and treatment
    status `t`.

    :param y: (n,) np.array containing outcomes
    :param t: (n,) np.array (bool) containing treatment status
    :return: float scalar representing estimated treatment effect
    """
    return y[t].mean() - y[~t].mean()


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


@njit
def _retrieve_index(index, index_sorted, split_index):
    """
    Given `index` (bool index of length of the original training data (n)),
    `index_sorted` (index of length of sum of True values in `index`) and
    `split_index` (index of `index_sorted` where the split should occur),
    computes new index left and new index right corresponding to the split.

    :param index: (n,) np.array (bool)
    :param index_sorted:
    :param split_index:
    :return: three dim tuple
    """
    left = index_sorted[: (split_index + 1)]
    right = index_sorted[(split_index + 1) :]
    nonzero_index = np.nonzero(index)[0]

    n = len(index)

    left_index = np.full((n,), False)
    right_index = np.full((n,), False)
    left_index[nonzero_index[left]] = True
    right_index[nonzero_index[right]] = True
    # global_split_index = nonzero_index[index_sorted[split_index]]

    return left_index, right_index


@njit
def _compute_treatment_effect_raw(sum_1, n_1, sum_0, n_0):
    """

    Args:
        sum_1:
        n_1:
        sum_0:
        n_0:

    Returns:

    """
    return sum_1 / n_1 - sum_0 / n_0


@njit
def _compute_loss_raw_left(yy_transformed, i, te):
    """

    Args:
        yy_transformed:
        i:
        te:

    Returns:

    """
    return te ** 2 - 2 * te * yy_transformed[: (i + 1)].sum()


@njit
def _compute_loss_raw_right(yy_transformed, i, te):
    """

    Args:
        yy_transformed:
        i:
        te:

    Returns:

    """
    return te ** 2 - 2 * te * yy_transformed[(i + 1) :].sum()


@njit
def _find_optimal_split_observation_loop(
    splitting_indices, yy, yy_transformed, xx, tt, loss
):
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


def _find_optimal_split(y, t, x, index, min_leaf):
    """

    Args:
        y:
        t:
        x:
        index:
        metric:
        loss_weighting:
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

        (
            jloss,
            jsplit_value,
            jsplit_index,
        ) = _find_optimal_split_observation_loop(
            splitting_indices, yy, yy_transformed, xx, tt, loss
        )

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

    df_ctree = pd.DataFrame(ctree_array, columns=column_names)
    df_ctree[
        ["id", "left_child", "right_child", "level", "split_feat"]
    ] = df_ctree[
        ["id", "left_child", "right_child", "level", "split_feat"]
    ].astype(
        "Int64"
    )

    df_out = df_ctree.set_index("id").sort_index()
    return df_out


def predict_causaltree(ctree, x):
    """

    Args:
        ctree:
        x:

    Returns:

    """
    pass
