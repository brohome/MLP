import pandas as pd

class DataLoader:
    """
    Загрузчик данных для удобной итерации по батчам.
    """
    def __init__(self, x: pd.DataFrame, y: pd.Series, batch_size: int = 1):
        self.x = x.to_numpy()
        self.y = y.to_numpy()
        self.batch_size = batch_size
        self.n_samples = self.x.shape[0]

    def iterate_batches(self):
        for i in range(0, self.n_samples, self.batch_size):
            x_batch = self.x[i:i + self.batch_size]
            y_batch = self.y[i:i + self.batch_size]
            yield x_batch.T, y_batch.T