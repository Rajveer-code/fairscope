import numpy as np
import pytest

from fairscope.core._utils import (
    _as_1d_arrays,
    _check_binary_labels,
    _check_binary_values,
    _check_scores,
)


def test_as_1d_arrays_length_mismatch_raises():
    with pytest.raises(ValueError):
        _as_1d_arrays(np.array([1, 2, 3]), np.array([1, 2]))


def test_as_1d_arrays_rejects_2d():
    with pytest.raises(ValueError):
        _as_1d_arrays(np.zeros((2, 2)))


def test_check_binary_labels_requires_both_classes():
    with pytest.raises(ValueError):
        _check_binary_labels(np.ones(5, dtype=int))


def test_check_binary_labels_rejects_non_binary():
    with pytest.raises(ValueError):
        _check_binary_labels(np.array([0, 1, 2]))


def test_check_binary_values_allows_single_class():
    out = _check_binary_values(np.zeros(4, dtype=int))
    assert out.tolist() == [0, 0, 0, 0]


def test_check_scores_rejects_nan():
    with pytest.raises(ValueError):
        _check_scores(np.array([0.1, np.nan, 0.3]))


def test_check_binary_values_rejects_nan():
    with pytest.raises(ValueError):
        _check_binary_values(np.array([0.0, np.nan, 1.0]))
