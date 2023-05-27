"""Module defining different Neural Network Loss functions."""
import math

from typing import TYPE_CHECKING, Callable, List, TypeVar, Union

from mypy_extensions import DefaultNamedArg

if TYPE_CHECKING:
    Number = TypeVar("Number", int, float)
    Matrix = List[List["Number"]]
    Vector = List["Number"]
    Tensor = Union["Matrix", "Vector"]
    LossFunction = Callable[
        ["Vector", "Vector", DefaultNamedArg(bool, "derivative")],  # noqa: F821
        Union["Number", "Vector"],
    ]

def relu(z: Union["Number", "Tensor"], derivative: bool = False) -> Union["Number", "Tensor"]:
    """Rectified linear Unit activtion function.

    Args:
        z (Number, Tensor): Output vector of layer prior to activation
        derivative: Toggle derivative of ReLU for backpropagation

    Returns:
        Activated output
    """
    if isinstance(z, (int,float)):
        if derivative and z>0:
            return 1
        elif z>0:
            return z
        else:
            return 0
    elif isinstance(z,list):
        return [relu(x,derivative) for x in z]

def sigmoid(z: Union["Number", "Tensor"], derivative: bool = False) -> Union["Number", "Tensor"]:
    """Sigmoid activtion function.

    Args:
        z (Number, Tensor): Output vector of layer prior to activation
        derivative: Toggle derivative of Sigmoid for backpropagation

    Returns:
        Activated output
    """
    if isinstance(z, (int,float)):
        if derivative:
            return sigmoid(z)*(1-sigmoid(z))
        return 1/(1+math.exp(-1*z))
    elif isinstance(z,list):
        return [sigmoid(x,derivative) for x in z]
