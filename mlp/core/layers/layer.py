import numpy as np
from abc import ABC, abstractmethod

class __Layer(ABC):
    @abstractmethod
    def forward(self, x): ...
    
    @abstractmethod
    def backward(self, grad): ...