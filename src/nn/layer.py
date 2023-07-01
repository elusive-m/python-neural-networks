from abc import ABC
from abc import abstractmethod as method
from abc import abstractproperty as property

import numpy as np
import numpy.typing as npt


class Layer(ABC):
    @method
    def initialize(self, inputs: int) -> None:
        pass

    @method
    def forward(
        self,
        X: npt.NDArray[np.float64],
        is_training: bool = False,
    ) -> npt.NDArray[np.float64]:
        pass

    @method
    def backward(
        self,
        dY: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], ...]:
        pass

    @property
    def outputs(self) -> int:
        pass

    @property
    def parameters(self) -> tuple[npt.NDArray[np.float64], ...]:
        pass
