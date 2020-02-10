# Module to fit a causal tree on numpy arrays using numba
# TODO: see issues on github
from itertools import count

import numpy as np
import pandas as pd


def _compute_child_node_ids(parent_id):
    """
    TODO: Write docstring.

    :param parent_id:
    :return:
    """
    left = 2 * parent_id + 1
    right = left + 1

    return left, right


def _compute_potential_splitting_points(t, min_leaf):
    """
    Computes potential splitting point indices.

    :param t: (n, ) np.array (bool) treatment status sorted wrt to features
    :param min_leaf:
    :return: sequence of indices to be considered for a split
    """
    nn = len(t)
    if nn <= min_leaf:
        return range(0)

    tmp = np.where(np.cumsum(t) == min_leaf)[0]
    if tmp.size == 0:
        return range(0)
    else:
        left_treated = tmp[0]
    tmp = np.where(np.cumsum(~t) == min_leaf)[0]
    if tmp.size == 0:
        return range(0)
    else:
        left_untreated = tmp[0]
    left = max(left_treated, left_untreated)

    tmp = np.where(np.cumsum(np.flip(t)) == min_leaf)[0]
    if tmp.size == 0:
        return range(0)
    else:
        right_treated = tmp[0]
    tmp = np.where(np.cumsum(np.flip(~t)) == min_leaf)[0]
    if tmp.size == 0:
        return range(0)
    else:
        right_untreated = tmp[0]
    right = nn - 1 - max(right_treated, right_untreated)

    if left > right - 1:
        return range(0)
    else:
        return range(left, right - 1)


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


def _estimate_treatment_effect(y, t):
    """
    Estimates average treatment effect (ATE) using outcomes `y` and treatment
    status `t`.

    :param y: (n,) np.array containing outcomes
    :param t: (n,) np.array (bool) containing treatment status
    :return: float scalar representing estimated treatment effect
    """
    return y[t].mean() - y[~t].mean()


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
    global_split_index = nonzero_index[index_sorted[split_index]]

    return left_index, right_index, global_split_index


def _find_optimal_split(y, t, x, index, metric, loss_weighting, min_leaf):
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
    split_var = None
    split_value = None
    split_index = None
    loss = np.inf

    for j in range(p):
        # loop through features

        index_sorted = np.argsort(x[index, j])
        xx = x[index, j][index_sorted]
        yy = y[index][index_sorted]
        tt = t[index][index_sorted]

        yy_transformed = _transform_outcome(yy, tt)

        splitting_points = _compute_potential_splitting_points(tt, min_leaf)
        for i in splitting_points:
            # loop through observations

            left_te = _estimate_treatment_effect(yy[: (i + 1)], tt[: (i + 1)])
            right_te = _estimate_treatment_effect(yy[(i + 1) :], tt[(i + 1) :])

            left_loss = metric(yy_transformed[: (i + 1)], left_te)
            right_loss = metric(yy_transformed[(i + 1) :], right_te)

            global_loss = loss_weighting(
                left_loss, right_loss, i + 1, len(yy) - i - 1
            )
            if global_loss < loss:
                split_var = j
                split_value = xx[i]
                split_index = i
                loss = global_loss

    # check if any split has occured.
    if loss == np.inf:
        return None

    # create index of observations falling in left and right leaf, respectively
    index_sorted = np.argsort(x[index, split_var])
    left, right, split_index = _retrieve_index(
        index, index_sorted, split_index
    )

    return left, right, split_var, split_value


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

    column_names = [
        "id",
        "left_child",
        "right_child",
        "level",
        "split_feat",
        "split_value",
        "treat_effect",
    ]

    df_out = pd.DataFrame(columns=column_names)

    tmp = _find_optimal_split(
        y,
        t,
        x,
        index,
        func_params["metric"],
        func_params["weight_loss"],
        crit_params["min_leaf"],
    )

    if tmp is None or level == crit_params["max_depth"]:
        # if we do not split the node must be a leaf, hence we add the
        # treatment effect
        treat_effect = _estimate_treatment_effect(y[index], t[index])

        info = np.array(
            [nodeid, np.nan, np.nan, level, np.nan, np.nan, treat_effect,]
        )

        to_append = pd.Series(info, column_names)
        return df_out.append(to_append, ignore_index=True)
    else:
        left, right, split_feat, split_value = tmp

        leftid = next(id_params["counter"])
        rightid = next(id_params["counter"])

        info = np.array([nodeid, leftid, rightid, level])

        split_info = np.array([split_feat, split_value, np.nan])
        info = np.append(info, split_info)
        to_append = pd.Series(info, column_names)
        df_out = df_out.append(to_append, ignore_index=True)

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

        df_out = df_out.append(out_left, ignore_index=True)
        df_out = df_out.append(out_right, ignore_index=True)

        return df_out


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
    df_ctree = _fit_node(
        y=y,
        t=t,
        x=x,
        index=index,
        crit_params=crit_params,
        func_params=func_params,
        id_params=id_params,
    )

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
