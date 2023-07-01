from abc import ABC
from abc import abstractmethod as method

import numpy as np
import numpy.typing as npt


class Optimizer(ABC):
    @method
    def initialize(self, parameters: list[npt.NDArray[np.float64]]) -> None:
        pass

    @method
    def update(
        self,
        parameters: list[npt.NDArray[np.float64]],
        gradients: list[npt.NDArray[np.float64]],
    ) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.05) -> None:
        self._lr = learning_rate

    def initialize(self, parameters: list[npt.NDArray[np.float64]]) -> None:
        pass

    def update(
        self,
        parameters: list[npt.NDArray[np.float64]],
        gradients: list[npt.NDArray[np.float64]],
    ) -> None:
        for i, gradient in enumerate(gradients):
            parameters[i] -= self._lr * gradient


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def initialize(self, parameters: list[npt.NDArray[np.float64]]) -> None:
        self._m = [np.zeros_like(param) for param in parameters]
        self._v = [np.zeros_like(param) for param in parameters]
        self._t = 0

    def update(
        self,
        parameters: list[npt.NDArray[np.float64]],
        gradients: list[npt.NDArray[np.float64]],
    ) -> None:
        self._t += 1

        for i, gradient in enumerate(gradients):
            self._m[i] = self._beta1 * self._m[i] + (1 - self._beta1) * gradient
            self._v[i] = self._beta2 * self._v[i] + (1 - self._beta2) * (gradient**2)

            m_hat = self._m[i] / (1 - self._beta1**self._t)
            v_hat = self._v[i] / (1 - self._beta2**self._t)

            parameters[i] -= self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)
