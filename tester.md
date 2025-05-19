<a id="intermittent_forecast.croston"></a>

# intermittent\_forecast.croston

Croston's method for forecasting intermittent time series.

<a id="intermittent_forecast.croston.Croston"></a>

## Croston Objects

```python
class Croston(BaseForecaster)
```

A class for fitting and forecasting intermittent time series.

Croston's method deconstructs a time series into separate demand and
interval series. It applies a Simple Exponential Smoothing (`SES`)
algorithm to the both series, then reconstructs this back into a forecast.
This class provides a convenient way to use Croston's method for
forecasting. It provides a simple interface to fit the model to a time
series, and to generate forecasts.

When fitting, variants of Croston's can be selected, including
Syntetos-Boylan Approximation (`SBA`), Shale-Boylan-Johnston (`SBJ`), and
Teunter-Syntetos-Babai (`TSB`). Each allows the selection of the smoothing
factors, alpha and beta. If not specified, alpha and beta will be optimised
through minimising the error between the fitted time series and the
original time series. The error is specified by the `optimisation_metric`,
which defaults to the Mean Squared Error (`MSE`), but can also be set to
the Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), Mean Squared
Rate (`MSR`), or Periods in Stock (`PIS`).

<a id="intermittent_forecast.croston.Croston.fit"></a>

#### fit

```python
def fit(ts: TSInput,
        variant: str = "CRO",
        alpha: float | None = None,
        beta: float | None = None,
        optimisation_metric: str = "MSE") -> Croston
```

Fit the model to the time-series.

**Arguments**:

- `ts` _ArrayLike_ - Time series to fit the model to. Must be
  1-dimensional and contain at least two non-zero values.
- `variant` _str_ - The Croston variant to use. Options are "SBA",
  "SBJ", "TSB", or default "CRO". These correspond to
  Syntetos-Boylan Approximation, Shale-Boylan-Johnston,
  Teunter-Syntetos-Babai, and the original Croston method,
  respectively.
- `alpha` _float, optional_ - Demand smoothing factor in the range
  [0,1]. Values closer to 1 will favour recent demand. If not
  set, the value will be optimised. Defaults to None.
- `beta` _float, optional_ - Interval smoothing factor in the range
  [0,1]. Values closer to 1 will favour recent intervals.  If not
  set, the value will be optimised. Defaults to None.
- `optimisation_metric` _str, optional_ - Metric to use when optimising
  for alpha and beta. The selected metric is used when comparing
  the error between the time series and the fitted in-sample
  forecast. Defaults to 'MSE'.
  

**Returns**:

- `self` _Croston_ - Fitted model instance.

<a id="intermittent_forecast.croston.Croston.forecast"></a>

#### forecast

```python
def forecast(start: int, end: int) -> TSArray
```

Forecast the time series using the fitted parameters.

The forecast is computed by appending the out of sample forecast to the
fitted values.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.croston.Croston.get_fitted_model_result"></a>

#### get\_fitted\_model\_result

```python
def get_fitted_model_result() -> dict[str, Any]
```

Get the fitted results.

