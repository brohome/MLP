import numpy as np

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    """
    MSE Loss
    y_pred — предсказания, shape (batch_size, output_size)
    y_true — истинные значения, shape (batch_size, output_size)
    """
    loss = np.mean((y_pred - y_true) ** 2)
    grad = 2 * (y_pred - y_true) / y_true.shape[0]
    return loss, grad

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """
    Cross-entropy loss
    y_pred — вероятности после softmax, shape (batch_size, num_classes)
    y_true — one-hot вектор меток, shape (batch_size, num_classes)
    """
    # Предотвращение лог(0)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    grad = (y_pred - y_true) / y_true.shape[0]
    return loss, grad

