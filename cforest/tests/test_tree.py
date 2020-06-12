import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from cforest.tree import _compute_global_loss
from cforest.tree import _compute_valid_splitting_indices
from cforest.tree import _find_optimal_split
from cforest.tree import _find_optimal_split_inner_loop
from cforest.tree import _predict_row_causaltree
from cforest.tree import _retrieve_index
from cforest.tree import _transform_outcome
from cforest.tree import predict_causaltree

tt = [
    np.array([False]),
    np.array([False, False, False, False, True]),
    np.array([True, True, True, True, False]),
    np.array([False, False, True, True, True]),
    np.concatenate(
        (
            np.full((10,), False),
            np.array([True, False, True, False]),
            np.full((10,), True),
        )
    ),
]
min_leafs = [2] * 5


@pytest.mark.parametrize("t, min_leaf", zip(tt, min_leafs))
def test__compute_valid_splitting_indices_with_empty_output(t, min_leaf):
    out = _compute_valid_splitting_indices(t, min_leaf)
    assert_array_equal(out, np.arange(0))


tt = [
    np.concatenate(
        (
            np.full((10,), False),
            np.array([True, True, False, True, True, False, False]),
            np.full((10,), True),
        )
    )
]
min_leafs = [2]
out_expected = [np.array([11, 12, 13, 14])]


@pytest.mark.parametrize(
    "t, min_leaf, out_exp", zip(tt, min_leafs, out_expected)
)
def test__compute_valid_splitting_indices_with_output(t, min_leaf, out_exp):
    out = _compute_valid_splitting_indices(t, min_leaf)
    assert_array_equal(out, out_exp)


@pytest.fixture
def setup_retrieve_index_for_completeness():
    index = np.concatenate(
        (
            np.full((10,), False),
            np.array([True, True, False, True, True, False, False]),
            np.full((10,), True),
        )
    )
    x = np.array(
        [
            0.32956842,
            -0.55119603,
            -1.11740483,
            -0.26300451,
            -0.06686618,
            0.21236623,
            0.06182492,
            0.66415156,
            -0.19704692,
            0.41878558,
            0.58971691,
            -1.3248038,
            -0.55965504,
            -0.28713562,
        ]
    )
    sorted_subset_index = np.argsort(x)
    split_index = int(len(index) / 2)

    out = {
        "index": index,
        "sorted_subset_index": sorted_subset_index,
        "split_index": split_index,
    }
    return out


def test__retrieve_index_for_completeness(
    setup_retrieve_index_for_completeness,
):
    left, right = _retrieve_index(**setup_retrieve_index_for_completeness)
    combined = np.array(left, dtype=int) + np.array(right, dtype=int)

    assert_array_equal(
        setup_retrieve_index_for_completeness["index"],
        np.array(combined, dtype=bool),
    )


def test__retrieve_index_reverse_engineer_split_point():
    pass


def test__retrieve_index_reverse_engineer_index_sorted():
    pass


def test__compute_global_loss():
    n_1l, n_0l, n_1r, n_0r = 10, 10, 10, 10
    sum_1l, sum_0l, sum_1r, sum_0r = 10, 10, 10, 10
    y_transformed = np.array(20 * [-2, 2])
    i = 20

    result = _compute_global_loss(
        sum_1l=sum_1l,
        sum_0l=sum_0l,
        sum_1r=sum_1r,
        sum_0r=sum_0r,
        n_1l=n_1l,
        n_0l=n_0l,
        n_1r=n_1r,
        n_0r=n_0r,
        y_transformed=y_transformed,
        i=i,
        use_transformed_outcomes=True,
    )
    assert result == 160.0


def _create_data_for_splitting_tests(n):
    x = np.linspace(-1, 1, num=n)
    np.random.seed(2)
    t = np.array(np.random.binomial(1, 0.5, n), dtype=bool)
    y = np.repeat([-1, 1], int(n / 2))
    y = np.insert(y, int(n / 2), -1)
    y = y + 2 * y * t
    return x, t, y


def test__find_optimal_split_inner_loop():
    """Create 1 dim. data for which we know that the split must occur at x = 0.
    """
    nobs = 10001
    x, t, y = _create_data_for_splitting_tests(n=nobs)
    y_transformed = _transform_outcome(y, t)
    splitting_indices = _compute_valid_splitting_indices(t, min_leaf=2)

    result = _find_optimal_split_inner_loop(
        splitting_indices=splitting_indices,
        x=x,
        t=t,
        y=y,
        y_transformed=y_transformed,
        min_leaf=4,
        use_transformed_outcomes=True,
    )
    _, split_value, split_index = result

    # need to check if the algorithms needs to hit 0.0 for sure or only approx.
    assert abs(split_value) < 0.02
    # as above (check if we found an index very close to the middle)
    assert abs(split_index - nobs / 2) < 15


def test__find_optimal_split():
    """Create multi dimensional data for which we know that the split must
    occur almost surely at the first feature."""
    nobs = 10001  # number of observations
    k = 10  # number of unrelated features
    x, t, y = _create_data_for_splitting_tests(n=nobs)

    # no seed on purpose
    unrelated_features = np.random.normal(loc=0, scale=2, size=(nobs, k))
    X = np.hstack((x.reshape((-1, 1)), unrelated_features))
    index = np.full((nobs,), True)

    tmp = _find_optimal_split(
        X=X, t=t, y=y, index=index, min_leaf=4, use_transformed_outcomes=False
    )
    _, _, split_feat, _ = tmp
    assert split_feat == 0


ctree = pd.read_csv("cforest/tests/data/fitted_ctree__predict_row_test.csv")
ctree[["left_child", "right_child", "level", "split_feat"]] = ctree[
    ["left_child", "right_child", "level", "split_feat"]
].astype("Int64")
ctrees = [ctree] * 4
rows = [
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, 1]),
    np.array([-1, -1]),
]
expected = [0.0, -5.0, 2.0, 14.0]


@pytest.mark.parametrize("ctree, row, exp", zip(ctrees, rows, expected))
def test__predict_row_causaltree(ctree, row, exp):
    prediction = _predict_row_causaltree(ctree, row)
    assert prediction == exp


def test__predict_causaltree():
    x = np.array(rows)
    exp = np.array(expected)

    prediction = predict_causaltree(ctree, x)
    assert_array_equal(prediction, exp)
