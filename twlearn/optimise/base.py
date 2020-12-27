from abc import ABC, abstractmethod


class GradientOptimiserBase(ABC):

    def __init__(self, loss, params):
        self.loss = loss
        self.params = params

    @abstractmethod
    def optimise(self):
        return NotImplementedError

    @abstractmethod
    def step(self):
        return NotImplementedError

    @abstractmethod
    def calculate_gradients(self):
        return NotImplementedError
