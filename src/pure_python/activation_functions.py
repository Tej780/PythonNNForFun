"""Module defining different Neural Network Loss functions."""
import math
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    T = TypeVar("T", float, list)


def relu(z: "T", derivative: bool = False) -> "T":
    """Rectified linear Unit activtion function.

    Args:
        z (Number, Tensor): Output vector of layer prior to activation
        derivative: Toggle derivative of ReLU for backpropagation

    Returns:
        Activated output
    """
    if isinstance(z, (int, float)):
        if derivative and z > 0:
            return 1
        elif z > 0:
            return z
        else:
            return 0
    elif isinstance(z, list):
        return [relu(x, derivative) for x in z]


def sigmoid(z: "T", derivative: bool = False) -> "T":
    """Sigmoid activtion function.

    Args:
        z (Number, Tensor): Output vector of layer prior to activation
        derivative: Toggle derivative of Sigmoid for backpropagation

    Returns:
        Activated output
    """
    if isinstance(z, (int, float)):
        if derivative:
            return sigmoid(z) * (1 - sigmoid(z))
        return 1 / (1 + math.exp(-1 * z))
    elif isinstance(z, list):
        return [sigmoid(x, derivative) for x in z]
