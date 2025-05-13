"""Contains the base class used for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class BaseForecaster(ABC):
    """Base class for forecasting models."""

    def __init__(self) -> None:
        """Initialise the forecaster."""

    # @abstractmethod
    # def fit(
    #     self,
    #     ts: npt.NDArray[np.float64],
    #     *args,
    #     **kwargs,
    # ) -> BaseForecaster:
    #     """Fit the model to the time-series. Must be implemented."""

    @abstractmethod
    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        """Returns the forecast."""
