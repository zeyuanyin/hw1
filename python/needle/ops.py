"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return pow(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # a = node.inputs
        # return self.scalar * pow(a, self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / b**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:  # defaults to the last two axes
            return array_api.swapaxes(a, -2, -1)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        if self.axes is None:
            return transpose(out_grad, axes=(-2, -1))
        else:
            return transpose(out_grad, axes=self.axes)

        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # print(a.shape, "->", self.shape)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a = node.inputs[0]

        if len(a.shape) != len(self.shape):
            reshape_out_grad = out_grad.sum(
                tuple(range(0, len(self.shape) - len(a.shape)))
            )
        else:
            reshape_out_grad = out_grad  # reshape_out_grad.shape = a.shape

        axis = None
        for x, y in zip(a.shape, self.shape):
            if x != y:
                axis = a.shape.index(x)
                break

        if axis is None:
            return reshape_out_grad
        else:
            return reshape_out_grad.sum(axes=(axis,)).reshape(a.shape)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print(a.shape, "->", array_api.sum(a, self.axes).shape, "by axes", self.axes)
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.axes is None:
            return broadcast_to(out_grad, a.shape)
        else:
            new_shape = list(a.shape)

            if isinstance(self.axes, int):
                new_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    new_shape[axis] = 1

            new_shape = tuple(new_shape)
            reshaped_out_grad = reshape(out_grad, new_shape)

            return reshaped_out_grad.broadcast_to(a.shape)

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a, b = node.inputs
        res_a = out_grad @ b.transpose()
        res_b = a.transpose() @ out_grad

        def reverse_broadcast(x, y):  # x has the shape which is broadcasted from y
            if x.shape == y.shape:
                return x
            else:
                return x.sum(tuple(range(0, len(x.shape) - len(y.shape))))

        return reverse_broadcast(res_a, a), reverse_broadcast(res_b, b)

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # print(a.shape, out_grad.shape)
        # print(type(a),type(out_grad))
        return Tensor(1 / a.numpy()) * out_grad
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return exp(a) * out_grad
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.maximum(a, 0)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a = node.inputs[0].realize_cached_data()

        return out_grad * Tensor(a > 0)

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
