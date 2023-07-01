import numpy as np
import numpy.typing as npt

from .layer import Layer


class BatchNorm(Layer):
    def __init__(self, epsilon: float = 1e-9, momentum: float = 0.9) -> None:
        self._epsilon = epsilon
        self._momentum = momentum

    def initialize(self, inputs: int) -> None:
        self._beta = np.zeros((1, inputs))
        self._gamma = np.ones((1, inputs))

        self._running_mean = self._beta.copy()
        self._running_variance = self._running_mean.copy()

    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        if is_training:
            mean: npt.NDArray[np.float64] = np.mean(X, axis=0, keepdims=True)
            variance: npt.NDArray[np.float64] = np.var(X, axis=0, keepdims=True)

            self._running_mean = self.ema(self._running_mean, mean)
            self._running_variance = self.ema(self._running_variance, variance)

            X_mu = X - mean
            istddev = 1 / np.sqrt(variance + self._epsilon)
            X_normalized: npt.NDArray[np.float64] = X_mu * istddev

            self._cache = (X_mu, X_normalized, istddev)
        else:
            mean = self._running_mean
            variance = self._running_variance

            X_normalized = (X - mean) / np.sqrt(variance + self._epsilon)

        return self._gamma * X_normalized + self._beta

    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], ...]:
        X_mu, X_normalized, istddev = self._cache
        m, *_ = X_mu.shape

        dX_normalized = dY * self._gamma
        dbeta = np.sum(dY, axis=0, keepdims=True)
        dgamma = np.sum(dY * X_normalized, axis=0, keepdims=True)

        dX = (istddev / m) * (
            m * dX_normalized
            - np.sum(dX_normalized, axis=0)
            - X_normalized * np.sum(dX_normalized * X_normalized, axis=0)
        )

        return dX, dbeta, dgamma

    @property
    def outputs(self) -> int:
        return self._beta.size

    @property
    def parameters(self) -> tuple[npt.NDArray[np.float64], ...]:
        return (self._beta, self._gamma)

    def ema(
        self,
        running: npt.NDArray[np.float64],
        current: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return self._momentum * running + (1 - self._momentum) * current
