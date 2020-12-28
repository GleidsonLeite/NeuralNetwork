import numpy as np
from ActivationFunctions import tanh


class NeuralNetwork:
    def __init__(self, layers: list, learning_rate: float, momentum: float):
        assert len(layers) == 4, "The neural net should have 4 layers"
        self.layers = layers
        self.lr = learning_rate
        self.momentum = momentum

        self.reset()

    def reset(self):
        rg = np.random.default_rng(1)
        self.w1 = 2 * rg.random((self.layers[1], self.layers[0] + 1)) - 1
        self.w2 = 2 * rg.random((self.layers[2], self.layers[1] + 1)) - 1
        self.w3 = 2 * rg.random((self.layers[3], self.layers[2] + 1)) - 1

        self.mw1 = np.zeros_like(self.w1)
        self.mw2 = np.zeros_like(self.w2)
        self.mw3 = np.zeros_like(self.w3)

        self.bias = np.array([[-1]])

    def forward(self, x):
        self.v = x
        self.v1 = np.dot(self.w1, np.vstack((self.v, self.bias)))
        self.y1 = tanh(self.v1)
        self.v2 = np.dot(self.w2, np.vstack((self.y1, self.bias)))
        self.y2 = tanh(self.v2)
        self.v3 = np.dot(self.w3, np.vstack((self.y2, self.bias)))
        self.y3 = tanh(self.v3)
        return self.y3

    def backward(self, error):
        grad3 = error
        self.dw3 = self.lr * np.dot(
            grad3, np.transpose(np.vstack((self.y2, self.bias))))

        grad2 = np.multiply(
            tanh(self.v2, True),
            np.dot(np.transpose(self.w3[:, :self.layers[-2]]), grad3))
        self.dw2 = self.lr * np.dot(
            grad2, np.transpose(np.vstack((self.y1, self.bias))))

        grad1 = np.multiply(
            tanh(self.v1, True),
            np.dot(np.transpose(self.w2[:, :self.layers[-3]]), grad2))
        self.dw1 = self.lr * np.dot(
            grad1, np.transpose(np.vstack((self.v, self.bias))))

        self.w1 += self.dw1 + self.momentum * self.mw1
        self.w2 += self.dw2 + self.momentum * self.mw2
        self.w3 += self.dw3 + self.momentum * self.mw3

        self.mw1 = self.dw1
        self.mw2 = self.dw2
        self.mw3 = self.dw3
