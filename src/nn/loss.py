from abc import ABC
from abc import abstractmethod as method

import numpy as np
import numpy.typing as npt


class Loss(ABC):
    @method
    def forward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> np.float64:
        pass

    @method
    def backward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        pass


class MSE(Loss):
    def forward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> np.float64:
        return np.mean((Y - Yhat) ** 2, dtype=np.float64)

    def backward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return 2 * (Yhat - Y) / Yhat.size


class BinaryCrossEntropy(Loss):
    def __init__(self, epsilon: float = 1e-9) -> None:
        self._epsilon = epsilon

    def forward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> np.float64:
        Yhat = np.clip(Yhat, self._epsilon, 1 - self._epsilon)
        return -np.mean(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat), dtype=np.float64)

    def backward(
        self,
        Y: npt.NDArray[np.float64],
        Yhat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        Yhat = np.clip(Yhat, self._epsilon, 1 - self._epsilon)
        return -(Y / Yhat - (1 - Y) / (1 - Yhat))
