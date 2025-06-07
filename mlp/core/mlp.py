import numpy as np

class MLP:
    def __init__(self, layers: list, learning_rate: float = 0.1):
        self.layers = layers
        self.learning_rate = learning_rate
    

    def parameters(self):
        """
        Чтобы удобно передовать в оптимизаторы
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                params.append((layer.W, layer.dW))
            if hasattr(layer, 'b') and hasattr(layer, 'db'):
                params.append((layer.b, layer.db))
        return params
    
    def forward(self, x: np.ndarray): 
        """
        Здесь мы делаем прямой проход по всем слоям.
        """
        self.x = x
        for i in self.layers:
            self.x = i.forward(self.x)
        return self.x
        
    def backward(self, loss_grad):
        """
        loss_grad — это ∂L/∂a, градиент функции потерь по последнему выходу модели.
        Мы проходим назад по всем слоям и обновляем веса.
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
