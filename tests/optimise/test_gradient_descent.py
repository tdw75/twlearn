from twlearn.optimise.gradient_descent import GradientDescent, GradientDescentParams

import numpy as np
import pytest


@pytest.fixture
def params():
    return GradientDescentParams(
        learn_rate=0.05
    )


def loss(data, coeff):
    return np.sum((data * coeff) ** 2)


def test_loss():
    loss_val = loss(np.array([1, 1]), np.array([3, 2]))
    assert loss_val == 18


def test_derivative(params):
    opt = GradientDescent(loss, params)
    deriv = opt.calculate_gradients(1, 0)
    assert deriv == 0


def test_optimise(params):
    opt = GradientDescent(loss, params)
    data = np.array([1, 1])
    results = opt.optimise(data)
    assert (results.loss.round(2) == 0).all()
    assert (results.coefficients.round(2) == np.array([0, 0])).all()
