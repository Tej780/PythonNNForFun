"""Matrix transformation utilities based on Python Collections."""

from typing import TYPE_CHECKING, Callable, List, TypeVar, Union

if TYPE_CHECKING:
    Number = TypeVar("Number", int, float)
    Matrix = List[List[Number]]
    Vector = List[Number]
    Tensor = Union[Matrix, Vector]
    LossFunction = Callable[[Vector, Vector, bool], Union[Number, Vector]]


def tensor_addition(A: "Tensor", B: "Tensor") -> "Tensor":
    """Elementwise sum between two 2D matrices of the same shape.

    Args:
        A (Tensor): A Tensor
        B (Tensor): B Tensor

    Returns:
        (Tensor): Tensor with elementwise summed elements
    """
    assert len(A) == len(B)
    out = []
    for i in range(len(A)):
        ai = A[i]
        bi = B[i]
        if isinstance(ai, list) and isinstance(bi, list):
            assert len(ai) == len(bi)
            tmp = []
            for j in range(len(ai)):
                tmp.append(ai[j] + bi[j])
        else:
            tmp = ai + bi
        out.append(tmp)
    return out


def matvec(W: "Matrix", x: "Vector") -> "Vector":
    """Multiple 2D Matrix and column vector.

    output = Wx + b

    Args:
        W (Matrix): (Weight) Matrix
        x (Vector): Column vector

    Returns:
        (Vector): Matrix-vector product
    """
    Wx = []
    for row in W:
        assert len(row) == len(x)
        Wx.append(sum([row[i] * x[i] for i in range(len(x))]))
    return Wx


def transpose(A: "Matrix") -> "Matrix":
    """Transpose 2D Matrix.

    Args:
        A (Matrix): A matrix

    Return:
        (Matrix): Matrix transpose
    """
    def _initialise_zeros(n: int, m: int):
        return [[0] * m for _ in range(n)]

    n = len(A)
    m = len(A[0])
    out = _initialise_zeros(m, n)
    for i in range(n):
        for j in range(m):
            out[j][i] = A[i][j]
    return out


def scalar_multiply(alpha: "Number", v: "Tensor") -> "Tensor":
    """Multiply all elements of a N-D Tensor with a scalar value.

    Args:
        alpha: Scalar value
        v: Multidimensional Tensor

    Returns:
        Tensor with same shape as original,
            with each element multiplied by scalar
    """
    if isinstance(v, list):
        return [scalar_multiply(alpha, x) for x in v]
    else:
        return alpha * v


def hadamard(a: list, b: list):
    """Hadamard Product"""
    assert len(a) == len(b)
    acc = []
    for i in range(len(a)):
        acc.append(a[i] * b[i])

    return acc
