"""Tests for `loss_functions` module."""

from typing import TYPE_CHECKING, List, TypeVar

import pytest

from src import loss_functions as lf

if TYPE_CHECKING:
    Number = TypeVar("Number", int, float)
    Vector = List["Number"]

def test_no_derivative():
    y = [1,2]
    target0 = [1,2]
    target1 = [1,3]
    assert lf.MSE(y,target0,False)==0.0
    assert lf.MSE(y,target1,False)==0.5

def test_derivative():
    y = [1,2,3]
    target0 = [1,2,3]
    target1 = [1,3,5]
    assert lf.MSE(y,target0,True)==[0.0, 0.0, 0.0]
    assert lf.MSE(y,target1,True)==[0.0, -0.6666666666666666, -1.3333333333333333]

def test_empty():
    y = []
    with pytest.raises(AssertionError) as e:
        lf.MSE(y,y)

def test_different_sizes():
    y = [1]
    t = [1,2]
    with pytest.raises(AssertionError) as e:
        lf.MSE(y,t)
