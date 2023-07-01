import numpy as np
import numpy.typing as npt

from .layer import Layer


class FC(Layer):
    def __init__(self, units: int) -> None:
        self._units = units

    def initialize(self, inputs: int) -> None:
        # He initialization
        self._weights = np.random.standard_normal((inputs, self._units))
        self._weights *= np.sqrt(2 / inputs)

        self._biases = np.zeros((1, self._units))

    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        self._cache = X
        return (X @ self._weights) + self._biases

    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], ...]:
        dZ, X = dY, self._cache
        m, *_ = X.shape

        dW = (X.T @ dZ) / m
        db = np.mean(dZ, axis=0, keepdims=True)
        dX = dZ @ self._weights.T

        return dX, dW, db

    @property
    def outputs(self) -> int:
        return self._units

    @property
    def parameters(self) -> tuple[npt.NDArray[np.float64], ...]:
        return (self._weights, self._biases)
