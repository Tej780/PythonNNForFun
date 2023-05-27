"""Neural Network utilities based on Python Collections."""

import random
from typing import TYPE_CHECKING, Callable, List, Tuple, TypeVar, Union

import matrix_operations as mat
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


def init_layer(n: int, m: int) -> Tuple["Matrix", "Vector"]:
    """Initialise an empty layer with n imputs and m outputs.

    Args:
        n: Size of the input vector
        m: Size of the output vector

    Returns:
        (Tuple[Matrix, Vector]): Initialised weights (m x n) matrix
            and biases (m x 1) vector for the layer
    """
    W = []
    b = []
    for _ in range(m):
        b.append(random.random())
        row = []
        for _ in range(n):
            row.append(random.random())
        W.append(row)
    return W, b


def init_NN(*layers: int):
    """Initialise the weights and biases for the fully connected neural network.

    The network must contain at least two layers (input and output).

    Args:
        layers (int): The number of nodes within each layer.

    Returns:
        (List[Matrix], List[Vector]): Lists of weights and biases for the network.
    """
    assert len(layers) > 1
    Ws = []
    bs = []
    for dim in list(zip(layers[:-1], layers[1:])):
        W, b = init_layer(*dim)
        Ws.append(W)
        bs.append(b)
    return Ws, bs


def forward(
    x: "Vector", weights: List["Matrix"], biases: List["Vector"]
) -> List["Vector"]:
    """Forward pass through the neural network.

    Args:
        x: Input vector
        weights (List[Matrix]): Neural network weights
        biases (List[Vector]): Neural network biases

    Returns:
        (List[Vector]): Output for each layer in the network
    """

    xs = [x]
    for i in range(len(weights)):
        x = mat.matvec(weights[i], x)
        x = mat.tensor_addition(x, biases[i])
        xs.append(x)
    return xs


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


def dCdw(delta: "Vector", o_i: "Vector") -> "Matrix":
    """Calculate the derivative of the Cost function with respect to the weight matrix.

    Outer product of input from the previous layer with the error from the output
    dC/dx_ij = o_j*delta_i

    Args:
        delta: Output error vector based on the derivative of the Cost function.
        o_i: input to the layer

    Returns:
        (Matrix): Derivative of Cost with respect to the weights

    """
    delta_C = []
    for j in range(len(delta)):
        tmp = []
        for i in range(len(o_i)):
            tmp.append(o_i[i] * delta[j])
        delta_C.append(tmp)
    return delta_C


def backprop(
    weights: List["Matrix"],
    biases: List["Vector"],
    y: "Vector",
    target: "Vector",
    loss_fn: "LossFunction",
    learning_rate: float,
):
    """Propagate error back through network weights and biases.

    Args:
        weights (List[Matrix]): Weight matrices of the neural network to update
        biases (List[Vector], optional): Optional biases to update
        y (Vector): Network output vector
        target (Vector): Target vector
        loss_fn (LossFunction): Loss function with respect to which
         the gradients are calculated

    Returns:
        Tuple[List[Matrix],List[Vector]]: Updated weights and biases
    """
    new_weights = []
    new_biases = []
    dl = loss_fn(y[-1], target, derivative=True)
    for i in reversed(range(len(weights))):
        new_weight = mat.tensor_addition(
            mat.scalar_multiply(-1 * learning_rate, dCdw(dl, y[i])), weights[i]
        )
        new_bias = mat.tensor_addition(
            mat.scalar_multiply(-1 * learning_rate, dl), biases[i]
        )
        new_weights.append(new_weight)
        new_biases.append(new_bias)
        dl = mat.matvec(mat.transpose(weights[i]), dl)

    return list(reversed(new_weights)), list(reversed(new_biases))
