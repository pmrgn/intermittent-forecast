"""Exponential Smoothing for Time Series Forecasting."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt

from intermittent_forecast.base_forecaster import BaseForecaster


class FittedParams(TypedDict):
    """TypedDict for fitted parameters."""

    alpha: float
    beta: float
    ts_fitted: npt.NDArray[np.float64]


class TripleExponentialSmoothing(BaseForecaster):
    """Triple Exponential Smoothing."""

    def _fit(
        self,
        ts: npt.NDArray[np.float64],
        trend_type="add",
        seasonal_type="add",
        period=7,
    ):
        self.ts = ts
        self.n = len(self.ts)
        self.period = period
        self.trend_type = trend_type
        self.seasonal_type = seasonal_type

        lvl, b, s = self.initialise_arrays(
            ts,
            period,
            trend_type,
            seasonal_type,
        )

        params = self.opt_params
        lvl, b, s, smooth = self._model(params, lvl, b, s)
        self.components = {"level": lvl, "trend": b, "seasonal": s}

    def forecast(
        self,
        start: int,
        end: int,
    ) -> npt.NDArray[np.float64]:
        h = end - self.n + 1
        if h >= 1:
            if self.trend_type == "add":
                if self.seasonal_type == "add":
                    forecast = np.array(
                        [
                            lvl[-1] + i * b[-1] + s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                elif self.seasonal_type == "mul":
                    forecast = np.array(
                        [
                            (lvl[-1] + i * b[-1]) * s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
            if self.trend_type == "mul":
                if self.seasonal_type == "add":
                    forecast = np.array(
                        [
                            lvl[-1] * b[-1] ** i + s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                elif self.seasonal_type == "mul":
                    forecast = np.array(
                        [
                            lvl[-1] * b[-1] ** i * s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
            smooth = np.concatenate((smooth, forecast))
        return smooth[start : end + 1]

    @staticmethod
    def compute_exponential_smoothing(
        alpha: int,
        beta: int,
        gamma: int,
        ts: npt.NDArray[np.float64],
        lvl: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        s: npt.NDArray[np.float64],
        period: int,
    ):
        for i in range(1, len(lvl)):
            lvl[i] = alpha * (ts[i - 1] - s[i - 1]) + (1 - alpha) * (
                lvl[i - 1] + b[i - 1]
            )
            b[i] = beta * (lvl[i] - lvl[i - 1]) + (1 - beta) * b[i - 1]
            s[i + period - 1] = (
                gamma * (ts[i - 1] - lvl[i - 1] - b[i - 1])
                + (1 - gamma) * s[i - 1]
            )
        return lvl, b, s, lvl[:-1] + b[:-1] + s[:-period]

    def _opt(self, *args):
        """Return squared error between timeseries and smoothed array"""
        *_, smooth = self._model(*args)
        return sqeuclidean(smooth, self.ts)

    @staticmethod
    def initialise_arrays(
        ts: npt.NDArray[np.float64],
        period: int,
        trend_type: str,
        seasonal_type: str,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        m = period
        n = len(ts)
        lvl = np.zeros(n + 1)
        b = np.zeros(n + 1)
        s = np.zeros(n + m)
        lvl[0] = ts[:m].mean()
        if trend_type == "add":
            b[0] = (ts[m : 2 * m].sum() - ts[:m].sum()) / m**2
        if trend_type == "mul":
            b[0] = (ts[m : 2 * m].sum() / ts[:m].sum()) ** (1 / m)
        if seasonal_type == "add":
            s[:m] = ts[:m] - lvl[0]
        if seasonal_type == "mul":
            s[:m] = ts[:m] / lvl[0]
        return lvl, b, s

    def optimise_parameters(self):
        """Find optimal values for the smoothing factors"""
        lvl, b, s = self.initialise_arrays()
        alpha_init = 0.5 / self.period
        beta_init = 0.1 * alpha_init
        gamma_init = 0.05 * (1 - alpha_init)
        params = [alpha_init, beta_init, gamma_init]
        bounds = [(0, 0.5), (0, 0.5), (0, 0.5)]
        args = (lvl, b, s)
        optimised = minimize(self._opt, params, args=args, bounds=bounds)
        self.opt_params = optimised.x

    def predict(self, start, end):
        """Return a forecast between the specified start and end points

        Parameters
        ----------
        start : int
            Zero-indexed, point to begin forecasting
        end : int
            Index of final prediction

        Returns
        -------
        forecast : np.ndarray

        """
        lvl, b, s = self.initialise_arrays()
        params = self.opt_params
        lvl, b, s, smooth = self._model(params, lvl, b, s)
        self.components = {"level": lvl, "trend": b, "seasonal": s}
        h = end - self.n + 1
        if h >= 1:
            if self.trend_type == "add":
                if self.seasonal_type == "add":
                    forecast = np.array(
                        [
                            lvl[-1] + i * b[-1] + s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                elif self.seasonal_type == "mul":
                    forecast = np.array(
                        [
                            (lvl[-1] + i * b[-1]) * s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
            if self.trend_type == "mul":
                if self.seasonal_type == "add":
                    forecast = np.array(
                        [
                            lvl[-1] * b[-1] ** i + s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
                elif self.seasonal_type == "mul":
                    forecast = np.array(
                        [
                            lvl[-1] * b[-1] ** i * s[-(-i % self.period) - 1]
                            for i in range(1, h + 1)
                        ],
                    )
            smooth = np.concatenate((smooth, forecast))
        return smooth[start : end + 1]
