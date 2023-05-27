"""Module defining different Neural Network Loss functions."""

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

def MSE(
    y: "Vector", target: "Vector", derivative: bool = False
) -> Union[float, List[float]]:
    """Mean Square Error between network output "vector" and target "vector".

    Args:
        y (Vector): Network output vector
        target (Vector): Target vector
        derivative (bool): Flag to calculate derivate of
            loss function for backpropagation

    Returns:
        (Union[float,List[float]]): Either the loss scalar for a single vector,
            or the delta errors "vector" for backprop
    """
    assert len(y) == len(target)
    if derivative:
        return [(2 / len(y)) * (y[i] - target[i]) for i in range(len(y))]
    diff = [(target[i] - y[i]) ** 2 for i in range(len(y))]
    return sum(diff) / len(diff)