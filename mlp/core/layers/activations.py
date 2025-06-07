import numpy as np
from .layer import __Layer
"""
Здесь находятся функции активаций в виде классов которые наследуются от абстрактного класса Layer.
У каждого их них есть forward и backward.
"""
class ReLu(__Layer):
    def forward(self, x: np.ndarray):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output * (self.x > 0)
        return grad_input


class Tanh(__Layer):
    def forward(self, x: np.ndarray):
        self.x = x
        return np.tanh(x)

    def backward(self, grad_output):
        grad_input = grad_output * (1 - np.tanh(self.x)**2)
        return grad_input

class Sin(__Layer):
    def forward(self, x: np.ndarray):
        self.x = x
        return np.sin(x)

    def backward(self, grad_output):
        grad_input = grad_output * np.cos(self.x)
        return grad_input
