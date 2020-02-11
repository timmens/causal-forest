import numpy as np
import pytest
from numpy.testing import assert_array_equal

from algorithms.causaltree import _compute_valid_splitting_indices


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
min_leafs = [
    2,
    2,
    2,
    2,
    2,
]


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
