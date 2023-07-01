import numpy as np
import numpy.typing as npt

from .layer import Layer


class Activation(Layer):
    def initialize(self, inputs: int) -> None:
        self._outputs = inputs

    @property
    def outputs(self) -> int:
        return self._outputs

    @property
    def parameters(self) -> tuple[()]:
        return ()


class ReLU(Activation):
    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        self._cache = X
        return np.maximum(X, 0.0)

    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64]]:
        return ((self._cache > 0.0).astype(np.float64) * dY,)


class Sigmoid(Activation):
    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        self._cache = X
        return 1 / (1 + np.exp(-X))

    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64]]:
        A = self.forward(self._cache)
        return (A * (1 - A) * dY,)
