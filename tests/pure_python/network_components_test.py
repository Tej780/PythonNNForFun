"""Tests for `network_components` module."""

from src.pure_python import loss_functions as lf
from src.pure_python import network_components as nc


def test_init_layer():
    """Test the initialisation of a layer for a given dimension."""
    weights, biases = nc.init_layer(3, 4)
    assert len(weights) == 4
    assert len(weights[0]) == 3
    assert len(biases) == 4


def test_init_NN():
    """Test the initialisation of the network for a given set of layer dimensions."""
    weights, biases = nc.init_NN(4, 3, 1)
    assert len(weights) == 2
    assert len(weights[0]) == 3
    assert len(weights[0][0]) == 4
    assert len(biases) == 2
    assert len(biases[0]) == 3


def test_forward():
    """Test the forward pass through a network for a given input."""
    weights = [[[1, 1], [1, 1]]]
    biases = [[1, 1]]
    x = [1, 1]
    y = [[1, 1], [3, 3]]
    assert nc.forward(x, weights, biases) == y


def test_backprop():
    """Test the backpropagation for a given target vector and MSE loss."""
    weights = [[[1, 1], [1, 1]]]
    biases = [[1, 1]]
    y = [[1, 1], [3, 3]]
    target = [1, 1]
    new_weights, new_biases = nc.backprop(
        weights, biases, y, target, loss_fn=lf.MSE, learning_rate=0.1
    )
    assert new_weights == [[[0.8, 0.8], [0.8, 0.8]]]
    assert new_biases == [[0.8, 0.8]]
