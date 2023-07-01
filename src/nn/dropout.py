import numpy as np
import numpy.typing as npt

from .layer import Layer


class Dropout(Layer):
    def __init__(self, probability: float = 0.5) -> None:
        self._preservation_probability = 1 - probability

    def initialize(self, inputs: int) -> None:
        self._outputs = inputs

    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        if is_training:
            p = self._preservation_probability

            self._mask = np.random.binomial(1, p, size=X.shape) / p

            return X * self._mask

        return X

    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64]]:
        return (dY * self._mask,)

    @property
    def outputs(self) -> int:
        return self._outputs

    @property
    def parameters(self) -> tuple[()]:
        return ()
