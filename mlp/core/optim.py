import numpy as np

class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param, grad in self.parameters:
            param -= self.learning_rate * grad


class Adam:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # счётчик шагов

        # Инициализируем моменты для каждого параметра
        self.m = [np.zeros_like(param) for param, _ in self.parameters]
        self.v = [np.zeros_like(param) for param, _ in self.parameters]

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Коррекция смещения
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Обновление параметров
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
