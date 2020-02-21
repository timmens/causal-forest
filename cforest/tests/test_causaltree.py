import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from cforest.tree import _compute_valid_splitting_indices
from cforest.tree import _find_optimal_split_observation_loop
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


def test__find_optimal_split_observation_loop():
    # Simulate data for which we know that the split must occur at x = 0
    numsim = 10000
    x = np.linspace(-1, 1, num=numsim + 1)
    np.random.seed(2)
    t = np.array(np.random.binomial(1, 0.5, numsim + 1), dtype=bool)
    y = np.repeat([-1, 1], int(numsim / 2))
    y = np.insert(y, int(numsim / 2), -1)
    y = y + 2 * y * t
    y_transformed = _transform_outcome(y, t)
    splitting_indices = _compute_valid_splitting_indices(t, min_leaf=2)
    loss = np.inf

    result = _find_optimal_split_observation_loop(
        splitting_indices=splitting_indices,
        x=x,
        t=t,
        y=y,
        y_transformed=y_transformed,
        min_leaf=4,
    )

    _, split_value, split_index = result

    # need to check if the algorithms needs to hit 0.0 for sure or only approx.
    assert abs(split_value) < 0.02
    # as above (check if we find an index close to the middle)
    assert abs(split_index - numsim / 2) < 15


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
