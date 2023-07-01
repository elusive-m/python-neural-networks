from collections.abc import Generator

import numpy as np
import numpy.typing as npt
from tqdm import trange

from .layer import Layer
from .loss import Loss
from .optimizers import Optimizer


class Model:
    def __init__(
        self,
        inputs: int,
        layers: list[Layer],
        optimizer: Optimizer,
        loss: Loss,
    ) -> None:
        self.loss = loss
        self.layers = layers
        self.optimizer = optimizer

        for layer in self.layers:
            layer.initialize(inputs)
            inputs = layer.outputs

        self._parameters = list(self._get_parameters())
        self.optimizer.initialize(self._parameters)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        epochs: int,
        mini_batch_size: int | None = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> None:
        if shuffle:
            m, *_ = X.shape

            p = np.random.permutation(m)

            X = X[p]
            Y = Y[p]

        if mini_batch_size is not None:
            self._mini_batch_gradient_descent(X, Y, epochs, mini_batch_size, verbose)
        else:
            self._batch_gradient_descent(X, Y, epochs, verbose)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._forward(X, is_training=False)

    def _forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = True,
    ) -> npt.NDArray[np.float64]:
        for layer in self.layers:
            X = layer.forward(X, is_training)
        return X

    def _backward(self, dL: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
        gradients = []

        dA = dL
        for layer in reversed(self.layers):
            dA, *dparameters = layer.backward(dA)
            gradients.extend(dparameters)

        return gradients

    def _compute_gradients_and_loss(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
    ) -> tuple[list[npt.NDArray[np.float64]], np.float64]:
        Yhat = self._forward(X)

        dL = self.loss.backward(Y, Yhat)
        gradients = self._backward(dL)

        loss = self.loss.forward(Y, Yhat)
        return gradients, loss

    def _get_parameters(self) -> Generator[npt.NDArray[np.float64], None, None]:
        for layer in reversed(self.layers):
            yield from layer.parameters

    def _mini_batch_gradient_descent(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        epochs: int,
        mini_batch_size: int,
        verbose: bool,
    ) -> None:
        Xb, Yb = self._make_mini_batches(X, Y, mini_batch_size)

        with trange(epochs, disable=not verbose) as bar:
            for _i in bar:
                loss = 0.0

                for n, (x, y) in enumerate(zip(Xb, Yb, strict=True)):
                    gradients, batch_loss = self._compute_gradients_and_loss(x, y)

                    self.optimizer.update(self._parameters, gradients)

                    loss = (loss * n + float(batch_loss)) / (n + 1)

                bar.set_description(f"Loss: {loss:.3f}")

    def _make_mini_batches(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        mini_batch_size: int,
    ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
        m, *_ = X.shape
        batches = m // mini_batch_size

        Xb = []
        Yb = []
        for i in range(batches):
            start = i * mini_batch_size
            stop = start + mini_batch_size

            Xb.append(X[start:stop])
            Yb.append(Y[start:stop])

        if stop != m:
            Xb.append(X[stop:])
            Yb.append(Y[stop:])

        return Xb, Yb

    def _batch_gradient_descent(
        self,
        X: npt.NDArray[np.float64],
        Y: npt.NDArray[np.float64],
        epochs: int,
        verbose: bool,
    ) -> None:
        with trange(epochs, disable=not verbose) as bar:
            for _i in bar:
                gradients, loss = self._compute_gradients_and_loss(X, Y)

                self.optimizer.update(self._parameters, gradients)

                bar.set_description(f"Loss: {loss:.3f}")
