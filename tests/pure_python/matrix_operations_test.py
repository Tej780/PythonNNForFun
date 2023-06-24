"""Tests for `matrix_operations` module."""

import pytest

from src.pure_python import matrix_operations as mo


def test_tensor_addition():
    x0 = []
    x1 = [[1, 2], [3, 4]]
    x2 = [[1, 2, 3], [1, 2, 3]]
    x3 = [[1, 2], [1, 2], [1, 2]]

    assert mo.tensor_addition(x0, x0) == []
    assert mo.tensor_addition(x1, x1) == [[2, 4], [6, 8]]
    with pytest.raises(AssertionError) as _:
        mo.tensor_addition(x1, x2)
    with pytest.raises(AssertionError) as _:
        mo.tensor_addition(x1, x3)


def test_matvec():
    W0 = [[1, 2], [3, 4]]
    W1 = [[1, 2, 3], [3, 4, 5]]
    x = [1, 2]

    valid_out = mo.matvec(W0, x)
    assert valid_out == [5, 11]
    assert len(valid_out) == len(x)
    with pytest.raises(AssertionError) as _:
        mo.matvec(W1, x)


def test_transpose():
    W = [[1, 2], [3, 4]]
    assert mo.transpose(W) == [[1, 3], [2, 4]]
    assert mo.transpose([[1]]) == [[1]]
    with pytest.raises(AssertionError) as _:
        mo.transpose([])
    with pytest.raises(AssertionError) as _:
        mo.transpose([[]])


def test_scalar_multiply():
    x = [10, 9, 8]
    W = [[1, 2], [3, 4]]
    a = 3

    assert mo.scalar_multiply(3, W) == [[3, 6], [9, 12]]
    assert mo.scalar_multiply(-0.5, x) == [-5, -4.5, -4]
    assert mo.scalar_multiply(4, a) == 12


def test_hadamard():
    x1 = [1, 2]
    x2 = [3, 4]
    W0 = [[]]
    W1 = [[1, 2], [3, 4]]
    W2 = [[5, 4], [3, 2]]

    assert mo.hadamard(x1, x1) == [1, 4]
    assert mo.hadamard(x1, x2) == [3, 8]
    assert mo.hadamard(W1, W1) == [[1, 4], [9, 16]]
    assert mo.hadamard(W1, W2) == [[5, 8], [9, 8]]

    with pytest.raises(AssertionError) as _:
        assert mo.hadamard(W0, W0)
    with pytest.raises(AssertionError) as _:
        assert mo.hadamard(W0, x1)
