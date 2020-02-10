# Module to fit a causal tree on numpy arrays using numba
# TODO: see issues on github
from itertools import count

import numpy as np


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


def _find_optimal_split(
    y, t, x, index, metric, loss_weighting, min_leaf, level
):
    """
    Given the data (`y`, `t`, `x`) finds optimal splitting point in the
    feature space given the observations selected in `index` and a metric
    `metric`. Note that the algorithm is the standard implementation of the
    recursive binary splitting algorithm for decision trees.

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

    # store split information
    split_information = np.array(
        [split_var, split_value, split_index, loss, level, np.nan, 0]
    )

    return left, right, split_information


def _fit_node(
    y,
    t,
    x,
    index,
    metric,
    loss_weighting,
    min_leaf,
    max_depth,
    counter,
    level=0,
    nodeid=0,
):
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
    :param level:
    :param nodeid:
    :return:
    """
    out = np.array([]).reshape((-1, 7))
    tmp = _find_optimal_split(
        y, t, x, index, metric, loss_weighting, min_leaf, level
    )

    if tmp is None or level == max_depth:
        # if we do not split the node must be a leaf, hence we add the
        # treatment effect
        te = _estimate_treatment_effect(y[index], t[index])
        out = np.array(4 * [np.nan] + [level, te, nodeid]).reshape((-1, 7))
        return out
    else:
        left, right, split_information = tmp
        split_information[6] = nodeid
        out = np.concatenate((out, split_information.reshape((-1, 7))), axis=0)

        leftid = next(counter)
        rightid = next(counter)

        out_left = _fit_node(
            y,
            t,
            x,
            left,
            metric,
            loss_weighting,
            min_leaf,
            max_depth,
            counter,
            level + 1,
            leftid,
        )
        out_right = _fit_node(
            y,
            t,
            x,
            right,
            metric,
            loss_weighting,
            min_leaf,
            max_depth,
            counter,
            level + 1,
            rightid,
        )

        if out_left.size != 0:
            out = np.concatenate((out, out_left), axis=0)
        if out_right.size != 0:
            out = np.concatenate((out, out_right), axis=0)

        return out


def fit_causaltree(
    y, t, x, min_leaf=3, max_depth=25, metric=None, loss_weighting=None
):
    """
    Wrapper function for core function _fit.

    :param y: (n,) np.array containing dependent variable
    :param t: (n,) np.array containing treatment status
    :param x: (n,p) np.array containing independent variables
    :param min_leaf: minimum number of either treated or untreated observations
                    to be in a leaf after the split
    :param max_depth: maxmimum number of allowed levels
    :param metric: a distance function, e.g. l2 loss
    :param loss_weighting: a function which computes the global loss of a split
                        by combining the losses in each subset using weights
                        given by the no. of obersavtions falling in the
                        respective subset.
    :return:
    """
    n = len(y)
    index = np.full((n,), True)

    if metric is None:

        def metric(outcomes, estimate):
            return np.sum((outcomes - estimate) ** 2)

    if loss_weighting is None:

        def loss_weighting(left_loss, right_loss, w1, w2):
            return left_loss + right_loss

    # initialize counter object
    counter = count()
    rootid = next(counter)

    # fit tree
    tree = _fit_node(
        y,
        t,
        x,
        index,
        metric,
        loss_weighting,
        counter,
        min_leaf,
        max_depth,
        0,
        rootid,
    )

    return tree


def _does_node_have_children(ctree, node_id):
    """
    Returns `True` if node `node_id` has children in tree `ctree` and False
    else.

    :param ctree:
    :param node_id:
    :return:
    """
    left, _ = _compute_child_node_ids(node_id)
    return np.isin(ctree[:, 6], left)


def predict_causaltree(ctree, x):
    """
    Predicts new outcomes using a fitted causal tree `ctree` from new features
    `x`.

    :param ctree: a fitted causal tree
    :param x: new covariates from which we want to predict outcomes
    :return: predicted outcomes
    """
    node = 0

    has_children = _does_node_have_children(ctree, node)
    while has_children:
        var, value = ctree[node, 0], ctree[node, 1]
        if x[var] <= value:
            node, _ = _compute_child_node_ids(node)
        else:
            _, node = _compute_child_node_ids(node)
        has_children = _does_node_have_children(ctree, node)

    out = ctree[node, 5]
    return out
