from itertools import count

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from algorithms.causaltree import _compute_valid_splitting_indices
from algorithms.causaltree import _find_optimal_split_observation_loop
from algorithms.causaltree import _retrieve_index
from algorithms.causaltree import _transform_outcome

# random number seed counter
counter = count(0)

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


def test__retrieve_index_for_completeness():
    seed = next(counter)
    np.random.seed(seed)

    index = tt[0]
    x = np.random.randn(len(index))
    sorted_subset_index = np.argsort(x)
    split_index = int(len(index) / 2)

    left, right = _retrieve_index(index, sorted_subset_index, split_index)
    combined = np.array(left, dtype=int) + np.array(right, dtype=int)

    assert assert_array_equal(index, np.array(combined, dtype=bool))


def test__retrieve_index_reverse_engineer_split_point():
    pass


def test__retrieve_index_reverse_engineer_index_sorted():
    pass


def test__find_optimal_split_observation_loop():
    # Simulate data for which we know that the split must occur at x = 0
    numsim = 10000
    x = np.linspace(-1, 1, num=numsim + 1)
    seed = next(counter)
    np.random.seed(seed)
    t = np.array(np.random.binomial(1, 0.5, numsim + 1), dtype=bool)
    y = np.repeat([-1, 1], int(numsim / 2))
    y = np.insert(y, int(numsim / 2), -1)
    y = y + 2 * y * t
    y_transformed = _transform_outcome(y, t)
    splitting_indices = _compute_valid_splitting_indices(t, min_leaf=2)
    loss = np.inf

    result = _find_optimal_split_observation_loop(
        splitting_indices, y, y_transformed, x, t, loss
    )

    _, split_value, split_index = result

    # need to check if the algorithms needs to hit 0.0 for sure or only approx.
    assert abs(split_value) < 0.02
    # as above (check if we find an index close to the middle)
    assert abs(split_index - numsim / 2) < 15
