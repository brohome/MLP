import numpy as np

from .layer import __Layer

class Dense(__Layer):
    """
    Это класс одного полносвязного слоя.
    """
    def __init__(self, in_features, out_features):
        self.W  = np.random.randn(out_features, in_features) * 0.01
        self.b  = np.zeros((out_features, 1))
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray):
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad_output: np.ndarray):
        # 1) градиент по входу
        grad_input = self.W.T @ grad_output

        # 2) градиенты по параметрам
        batch_size = grad_output.shape[1]
        self.dW = (grad_output @ self.x.T) / batch_size
        self.db = np.sum(grad_output, axis=1, keepdims=True) / batch_size

        return grad_input
