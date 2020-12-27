from dataclasses import dataclass

import numpy as np

from twlearn.optimise.base import GradientOptimiserBase


@dataclass
class GradientDescentParams:
    learn_rate: float


@dataclass
class GradientDescentResults:
    coefficients: np.ndarray
    loss: float


class GradientDescent(GradientOptimiserBase):

    def __init__(self, loss, params: GradientDescentParams):
        super(GradientDescent, self).__init__(loss, params)

    def step(self, coeff: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return coeff - self.params.learn_rate * gradient

    def optimise(self, data: np.ndarray, coeff_start: np.ndarray = None) -> GradientDescentResults:
        coeff = coeff_start or np.random.random(data.shape)
        loss = self.loss(data, coeff)

        for i in range(100):
            grads = self.calculate_gradients(data, coeff)
            coeff = self.step(coeff, grads)
            loss = self.loss(data, coeff)

        return GradientDescentResults(coefficients=coeff, loss=loss)

    def calculate_gradients(self, data: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        # initially hardcoded for testing purposes
        # TODO: calculate with backpropagation
        return 2 * data * coeff
