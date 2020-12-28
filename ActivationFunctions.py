import numpy as np


# activation function sigmoid
def sigmoid(x, derivate=False):
    function = lambda v: np.power(1 + np.exp(-v), -1)
    dFunction = lambda v: function(v) * (1 - function(v))
    return dFunction(x) if derivate else function(x)


# activation function tanh
def tanh(x, derivate=False):
    a = 1.7159
    b = 2 / 3
    function = lambda v: a * np.tanh(b * v)
    dFunction = lambda v: 4 * a * b / np.power(
        np.exp(-b * x) + np.exp(b * x), 2)
    return dFunction(x) if derivate else function(x)
