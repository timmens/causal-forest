# Module to fit a causal tree on numpy arrays using numba
# TODO:
#  1) Check if it works
#  2) numba implementation

import numpy as np


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

    if left > right:
        return range(0)
    else:
        return range(left, right)


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


def _retrieve_index(index, index_sorted, split_index):
    """
    Given `index` (bool index of length of the original training data (n)),
    `index_sorted` (index of length of sum of True values in `index`) and
    `split_index` (index of `index_sorted` where the split should occur),
    computes new index left and new index right corresponding to the split.
    TODO:
        1) check if it should be index_sorted[:split_index] or
            index_sorted[:(split_index + 1)]

    :param index: (n,) np.array (bool)
    :param index_sorted:
    :param split_index:
    :return: three dim tuple
    """
    left = index_sorted[:(split_index + 1)]
    right = index_sorted[(split_index + 1):]
    nonzero_index = np.nonzero(index)[0]

    n = len(index)

    left_index = np.full((n, ), False)
    right_index = np.full((n, ), False)
    left_index[nonzero_index[left]] = True
    right_index[nonzero_index[right]] = True
    global_split_index = nonzero_index[index_sorted[split_index]]

    return left_index, right_index, global_split_index


def _find_optimal_split(y, t, x, index, metric, min_leaf, level):
    """
    Given the data (`y`, `t`, `x`) finds optimal splitting point in the
    feature space given the observations selected in `index` and a metric
    `metric`. Note that the algorithm is very similar to finding a minimum.
    TODO: 1) computation of global loss as convex combination?

    :param y: (n,) np.array containing dependent variable
    :param t: (n,) np.array (bool) containing treatment status
    :param x: (n,p) np.array containing independent variables
    :param index: (n,) np.array (bool) representing observations to consider
    :param metric: a distance function, e.g. l2 loss
    :param min_leaf: minimum number of either treated or untreated observations
                    to be in a leaf after the split
    :return:
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

            left_te = _estimate_treatment_effect(yy[:(i+1)], tt[:(i+1)])
            right_te = _estimate_treatment_effect(yy[(i+1):], tt[(i+1):])

            left_loss = metric(yy_transformed, left_te)
            right_loss = metric(yy_transformed, right_te)

            global_loss = left_loss + right_loss
            if global_loss < loss:
                split_var = j
                split_value = xx[i]
                split_index = i
                loss = global_loss

    # check if any split has occured
    if loss == np.inf:
        return None

    # create index of observations falling in left and right leaf, respectively
    index_sorted = np.argsort(x[index, split_var])
    left, right, split = _retrieve_index(index, index_sorted, split_index)

    # store split information
    split_information = np.array([split_var, split_value, split, loss, level])

    return left, right, split_information


def _fit(y, t, x, index, metric, min_leaf, level=0):
    """
    Core function which recursively splits the feature space until stopping
    criterium is reached (`min_leaf`) and returns splitting information.
    Fits a Causal Tree on the data (`y`, `t`, `x`), where `y` and `t` are
    1d np.arrays of length n and `x` is a 2d np.array of dimension n x p.

    :param y: (n,) np.array containing dependent variable
    :param t: (n,) np.array containing treatment status
    :param x: (n,p) np.array containing independent variables
    :param index: (n,) np.array (bool) containing indices of observations
    :param metric: a distance function, e.g. l2 loss
    :param min_leaf: minimum number of either treated or untreated observations
                    to be in a leaf after the split
    :return:
    """
    out = np.array([]).reshape((-1, 5))
    tmp = _find_optimal_split(y, t, x, index, metric, min_leaf, level)
    if tmp is None:
        return out
    else:
        left, right, split_information = tmp
        out = np.concatenate((out, split_information.reshape((1, -1))), axis=0)
        level_left = level + 1
        level_right = level + 1

        out_left = _fit(y, t, x, left, metric, min_leaf, level_left)
        out_right = _fit(y, t, x, right, metric, min_leaf, level_right)

        if out_left.size != 0:
            out = np.concatenate((out, out_left), axis=0)
        if out_right.size != 0:
            out = np.concatenate((out, out_right), axis=0)

        return out


def fitcausaltree(y, t, x, metric, min_leaf):
    """
    Wrapper function for core function _fit.

    :param y: (n,) np.array containing dependent variable
    :param t: (n,) np.array containing treatment status
    :param x: (n,p) np.array containing independent variables
    :param metric: a distance function, e.g. l2 loss
    :param min_leaf: minimum number of either treated or untreated observations
                    to be in a leaf after the split
    :return:
    """
    n = len(y)
    index = np.full((n,), True)

    tree = _fit(y, t, x, index, metric, min_leaf)

    return tree
