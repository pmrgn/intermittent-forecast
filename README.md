# Forecasting Intermittent Time Series 

This package contains tools for forecasting intermittent time series, such using Croston's method or one of its variants - SBA, SBJ, and TSB. In addition, the Aggregate-Disaggregate Intermittent Demand Approach can be used with the `ADIDA` class. This allows for the removal of intermittency within the data so that other forecasting models can be utilised - such as Simple, Double and Triple Exponential Smoothing.

Each of the forecasting models has the ability to optimise for parameters when using the `fit` method. Error metrics used for optimisation can be chosen, such as selecting between the MSE, MAE, MSR and Periods in Stock (PIS).

See the documentation below for examples on using this package.

## Installation

The intermittent-forecast package is hosted on PyPI and can be installed using pip.

    pip install intermittent-forecast

Alternatively, you can clone the repo

	git clone git@github.com:pmrgn/intermittent-forecast.git

## Documentation

* [intermittent\_forecast.aggregators](#intermittent_forecast.aggregators)

  * [ADIDA](#intermittent_forecast.aggregators.adida.ADIDA)
* [intermittent\_forecast.forecasters](#intermittent_forecast.forecasters)

  * [Croston](#intermittent_forecast.forecasters.croston.Croston)

  * [SimpleExponentialSmoothing](#intermittent_forecast.forecasters.simple_exponential_smoothing.SimpleExponentialSmoothing)

  * [DoubleExponentialSmoothing](#intermittent_forecast.forecasters.double_exponential_smoothing.DoubleExponentialSmoothing)

  * [TripleExponentialSmoothing](#intermittent_forecast.forecasters.triple_exponential_smoothing.TripleExponentialSmoothing)



<a id="intermittent_forecast.aggregators"></a>

# `intermittent_forecast.aggregators`

<a id="intermittent_forecast.aggregators.adida.ADIDA"></a>

## `ADIDA`

```python
class ADIDA(
  aggregation_period: int,
  aggregation_mode: str,
  disaggregation_mode: str,
)
```

Aggregate-Disaggregate Intermittent Demand Approach (ADIDA).

ADIDA is a forecasting methodology designed for handling intermittent time
series. The approach helps improve forecast accuracy by transforming the
problem into a more stable one via temporal aggregation.

The method involves aggregating a high-frequency time series (e.g., daily
observations) into a lower-frequency series (e.g., weekly) as a means to
reduce variability. This allows for a range of forecasting models to be
applied to the aggregated series, e.g. Exponential Smoothing. Once a model
has been fit to the aggregated series, the disaggregation process is
performed to return it to the original time series.

**Arguments**:

- `aggregation_period` _int_ - Number of time periods to aggregate. E.g,
  aperiod of 7 would be used to aggregate a daily time series into a
  weekly time series.
- `aggregation_mode` _str_ - The aggregation mode, either "sliding" or
  "block". A sliding window will aggregate the time series by moving
  the window one time period at a time. A block window will aggregate
  the time series by moving the window one block at a time. the time
  series by moving the window one block at a time.
- `disaggregation_mode` _str_ - The disaggregation mode, either
  "seasonal" or"uniform". A seasonal disaggregation will disaggregate
  the time series by calculating the proportion of demand for each
  step in the cycle. A uniform disaggregation will disaggregate the
  time series by evenly distributing the demand across the cycle.
  

**Example**:
```python
  >>> # Example intermittent time series.
  >>> ts = [
  ...     3, 0, 0, 4, 0, 0, 0,
  ...     1, 0, 5, 1, 1, 0, 0,
  ...     0, 0, 0, 8, 3, 0, 1,
  ...     0, 1, 0, 4, 3, 0, 0,
  ... ]
  
  >>> # Initialise ADIDA model.
  >>> from intermittent_forecast.aggregators import ADIDA
  >>> adida = ADIDA(
  ...     aggregation_period=7,
  ...     aggregation_mode="block",
  ...     disaggregation_mode="seasonal",
  ... )
  
  >>> # Import a forecasting model to use on the aggregated series.
  >>> from intermittent_forecast import forecasters
  
  >>> # Fit using ADIDA, passing in an instance of the forecasting model.
  >>> # Any valid keyword arguments used by the model can be passed in,
  >>> # e.g. alpha for SimpleExponentialSmoothing.
  >>> adida = adida.fit(
  ...     model=forecasters.SimpleExponentialSmoothing(),
  ...     ts=ts,
  ...     alpha=0.3,
  ... )
  
  >>> # Forecast the next 7 periods.
  >>> adida.forecast(start=len(ts), end=len(ts)+7)
  array([0.97108571, 0.24277143, 1.21385714, 4.12711429, 1.6994    ,
  0.        , 0.24277143, 0.97108571])
```
<a id="intermittent_forecast.aggregators.adida.ADIDA.fit"></a>

#### `fit`

```python
def fit(
  model: forecaster,
  ts: ArrayLike, 
  **kwargs: Any,
) -> ADIDA
```

Aggregate the time series and fit using the forecasting model.

**Arguments**:

- `model` _forecaster_ - Forecasting model to use on the
  aggregated time series, which can be any of the forecaster
  class instances, e.g. Croston, SimpleExponentialSmoothing.
- `ts` _ArrayLike_ - Time series to fit.
- `**kwargs` _Any_ - Additional keyword arguments to pass to the
  forecasting model. Refer to the documentation for the `fit`
  method of the forecasting model you are using for valid keyword
  arguments.
  

**Returns**:

- `self` _ADIDA_ - Fitted model instance.

<a id="intermittent_forecast.aggregators.adida.ADIDA.forecast"></a>

#### `forecast`

```python
def forecast(start: int, end: int) -> np.ndarray
```

Forecast the time series using the ADIDA method.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.aggregators.adida.ADIDA.get_fit_result"></a>

#### `get_fit_result`

```python
def get_fit_result() -> dict[str, Any]
```

Return the a dictionary of results if model has been fit.

<a id="intermittent_forecast.forecasters"></a>

# `intermittent_forecast.forecasters`

<a id="intermittent_forecast.forecasters.croston.Croston"></a>

## `Croston`

```python
class Croston()
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

**Example**:
```python
  >>> # Initialise an instance of Croston, fit a time series and create
  >>> # a forecast.
  >>> from intermittent_forecast.forecasters import Croston
  >>> ts = [0, 3, 0, 4, 0, 0, 0, 2, 0]
  >>> cro = Croston().fit(ts=ts, alpha=0.5, beta=0.2)
  >>> cro.forecast(start=0, end=10)
  array([       nan,        nan, 1.125     , 1.125     , 1.38157895,
  1.38157895, 1.38157895, 1.38157895, 0.97287736, 0.97287736])
  
  >>> # Smoothing parameters can instead be optimised with a chosen
  >>> # error metric.
  >>> cro = Croston().fit(ts=ts, optimisation_metric="MSR")
  >>> cro.forecast(start=0, end=10)
  array([       nan,        nan, 1.125     , 1.125     , 1.18876368,
  1.18876368, 1.18876368, 1.18876368, 1.05619218, 1.05619218])
  
  >>> # Access a dict of the fitted values, get smoothing parameter beta.
  >>> result = cro.get_fit_result()
  >>> result["beta"]
  0.2145546005097181
```
<a id="intermittent_forecast.forecasters.croston.Croston.fit"></a>

#### `fit`

```python
def fit(
  ts: ArrayLike,
  variant: str = "CRO",
  alpha: float | None = None,
  beta: float | None = None,
  optimisation_metric: str = "MSE",
) -> Croston
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

<a id="intermittent_forecast.forecasters.croston.Croston.forecast"></a>

#### `forecast`

```python
def forecast(start: int, end: int) -> np.ndarray
```

Forecast the time series using the fitted parameters.

The forecast is computed by appending the out of sample forecast to the
fitted values.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.forecasters.croston.Croston.get_fit_result"></a>

#### `get_fit_result`

```python
def get_fit_result() -> dict[str, Any]
```

Return a dictionary of results if model has been fit.




<a id="intermittent_forecast.forecasters.simple_exponential_smoothing.SimpleExponentialSmoothing"></a>

## `SimpleExponentialSmoothing`

```python
class SimpleExponentialSmoothing()
```

A class for forecasting time series using Simple Exponential Smoothing.

Simple Exponential Smoothing (`SES`) is a time series forecasting method
for data that does not exhibit trend or seasonal patterns. It
applies a weighted average to past observations, where the weights decay
exponentially with time.

When fitting, the smoothing factor alpha can be specified manually or be
automatically optimised. If not provided, alpha is optimised by minimising
the error between the fitted and actual values of the time series. The
error minimization is based on a chosen `optimisation_metric`, which
defaults to the Mean Squared Error (`MSE`). Other available metrics include
Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), and Mean Squared
Rate (`MSR`).

**Example**:
```python
  >>> # Initialise an instance of SimpleExponentialSmoothing, fit a time
  >>> # series and create a forecast.
  >>> from intermittent_forecast.forecasters import SimpleExponentialSmoothing
  >>> ts = [40, 28, 35, 41, 33, 21, 37, 20]
  >>> ses = SimpleExponentialSmoothing().fit(ts=ts, alpha=0.3)
  >>> ses.forecast(start=0, end=8)
  array([40.       , 40.       , 36.4      , 35.98     , 37.486    ,
  36.1402   , 31.59814  , 33.218698 , 29.2530886])
  
  >>> # Smoothing parameters can instead be optimised with a chosen
  >>> # error metric.
  >>> ses = SimpleExponentialSmoothing()
  >>> ses = ses.fit(ts=ts, optimisation_metric="MSR")
  >>> ses.forecast(start=0, end=8)
  array([40.        , 40.        , 36.1204974 , 35.75824969, 37.45286502,
  36.0132899 , 31.15961513, 33.04776416, 28.82952791])
  
  >>> # Access a dict of the fitted values, get smoothing parameter alpha.
  >>> result = ses.get_fit_result()
  >>> result["alpha"]
  0.32329188326949737
```
<a id="intermittent_forecast.forecasters.simple_exponential_smoothing.SimpleExponentialSmoothing.fit"></a>

#### `fit`

```python
def fit(
  ts: ArrayLike,
  alpha: float | None = None,
  optimisation_metric: str | None = None,
) -> SimpleExponentialSmoothing
```

Fit the model to the time-series.

**Arguments**:

- `ts` - ArrayLike
  Time series to fit the model to. Must be 1-dimensional and
  contain at least two non-zero values. If using multiplicative
  smoothing, the time series must be entirely positive.
- `alpha` - float, optional
  Level smoothing factor in the range [0, 1]. Values closer to 1
  will favour recent demand. If not set, the value will be
  optimised.
- `optimisation_metric` - {'MAR', 'MAE', 'MSE', 'MSR', 'PIS'},
  default='MSE' Metric to use when optimising for alpha and beta.
  The selected metric is used when comparing the error between
  the time series and the fitted in-sample forecast.
  

**Returns**:

- `self` - SimpleExponentialSmoothing
  Fitted model instance.

<a id="intermittent_forecast.forecasters.simple_exponential_smoothing.SimpleExponentialSmoothing.forecast"></a>

#### `forecast`

```python
def forecast(start: int, end: int) -> np.ndarray
```

Forecast the time series using the fitted parameters.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.forecasters.simple_exponential_smoothing.SimpleExponentialSmoothing.get_fit_result"></a>

#### `get_fit_result`

```python
def get_fit_result() -> dict[str, Any]
```

Return the a dictionary of results if model has been fit.

<a id="intermittent_forecast.forecasters.double_exponential_smoothing.DoubleExponentialSmoothing"></a>

## `DoubleExponentialSmoothing`

```python
class DoubleExponentialSmoothing()
```

A class for forecasting time series using Double Exponential Smoothing.

Double Exponential Smoothing (`DES`), also known as Holt's linear method,
extends `Simple Exponential Smoothing` by incorporating a trend component.
It is suitable for time series that exhibit a linear trend but no
seasonality. The method applies exponential smoothing separately to the
level and the trend of the series.

This class provides a simple interface to fit the DES model to a time
series. The model uses two smoothing factors: `alpha` for the level
component and `beta` for the trend component. Both parameters can be
specified manually or optimised automatically. If not provided, they will
be selected by minimising the error between the fitted and actual time
series values. The `optimisation_metric` used for fitting defaults to
the Mean Squared Error (`MSE`), but can also be set to alternative metrics
such as Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), or Mean
Squared Rate (`MSR`).

**Example**:
```python
  >>> # Initialise an instance of DoubleExponentialSmoothing, fit a time
  >>> # series and create a forecast.
  >>> from intermittent_forecast.forecasters import DoubleExponentialSmoothing
  >>> ts = [12, 14, 16, 16, 15, 20, 22, 26]
  >>> des = DoubleExponentialSmoothing().fit(ts=ts, alpha=0.3, beta=0.1)
  >>> des.forecast(start=0, end=8) # In-sample forecast
  array([12.        , 14.        , 16.        , 18.        , 19.34      ,
  19.8478    , 21.707826  , 23.61860942, 26.22759953])
  
  >>> # Out of sample forecasts are constructed from the final level and
  >>> # trend values.
  >>> des.forecast(start=9, end=12)
  array([28.12217247, 30.01674541, 31.91131834, 33.80589128])
  
  >>> # Smoothing parameters can instead be optimised with a chosen
  >>> # error metric.
  >>> des = DoubleExponentialSmoothing()
  >>> des = des.fit(ts=ts, optimisation_metric="MSR")
  >>> des.forecast(start=0, end=8)
  array([12.        , 14.        , 16.        , 18.        , 18.52701196,
  17.19289471, 19.22500534, 22.26717482, 27.03666424])
  
  >>> # Access a dict of the fitted values.
  >>> result = des.get_fit_result()
  >>> result["alpha"], result["beta"]
  (0.36824701090945217, 1.0)
```
<a id="intermittent_forecast.forecasters.double_exponential_smoothing.DoubleExponentialSmoothing.fit"></a>

#### `fit`

```python
def fit(
  ts: ArrayLike,
  alpha: float | None = None,
  beta: float | None = None,
  optimisation_metric: str | None = None,
) -> DoubleExponentialSmoothing
```

Fit the model to the time-series.

**Arguments**:

- `ts` _ArrayLike_ - Time series to fit the model to. Must be
  1-dimensional and contain at least two non-zero values. If
  using multiplicative smoothing, the time series must be
  entirely positive.
- `alpha` _float, optional_ - Level smoothing factor in the range
  [0,1]. Values closer to 1 will favour recent demand. If not
  set, the value will be optimised.
- `beta` _float, optional_ - Trend smoothing factor in the range [0, 1].
  Values closer to 1 will favour recent demand. If not set, the
  value will be optimised.
- `optimisation_metric` _str, optional_ - Metric to use when optimising
  for alpha and beta. Options are 'MAR', 'MAE', 'MSE', 'MSR',
  'PIS'. Defaults to 'MSE'. The selected metric is used when
  comparing the error between the time series and the fitted
  in-sample forecast.
  

**Returns**:

- `self` _DoubleExponentialSmoothing_ - Fitted model instance.

<a id="intermittent_forecast.forecasters.double_exponential_smoothing.DoubleExponentialSmoothing.forecast"></a>

#### `forecast`

```python
def forecast(start: int, end: int) -> np.ndarray
```

Forecast the time series using the fitted parameters.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.forecasters.double_exponential_smoothing.DoubleExponentialSmoothing.get_fit_result"></a>

#### `get_fit_result`

```python
def get_fit_result() -> dict[str, Any]
```

Return the a dictionary of results if model has been fit.



<a id="intermittent_forecast.forecasters.triple_exponential_smoothing.TripleExponentialSmoothing"></a>

## `TripleExponentialSmoothing`

```python
class TripleExponentialSmoothing()
```

A class for forecasting time series using Triple Exponential Smoothing.

Triple Exponential Smoothing (`TES`), also referred to as Holt-Winters
Exponential Smoothing, extends Double Exponential Smoothing by
incorporating a seasonal component. It is designed for time series data
that exhibits both trend and seasonality.The method simultaneously smooths
the level, trend, and seasonal components of the series using exponential
weighting. This class provides an easy-to-use interface to fit the TES
model and to generate forecasts that capture both seasonal and trend
behavior over time.

The model uses three smoothing parameters: alpha (level), beta (trend), and
gamma (seasonality). These can be specified manually or optimised
automatically by minimising the difference between fitted and actual
values. Seasonal patterns can be either additive or multiplicative, and the
type should be selected based on the characteristics of the data.

The `optimisation_metric` defaults to Mean Squared Error (`MSE`), but can
also be set to Mean Absolute Error (`MAE`), Mean Absolute Rate (`MAR`), or
Mean Squared Rate (`MSR`), among others.

**Example**:
```python
  >>> # Initialise an instance of TripleExponentialSmoothing, fit a time
  >>> # series and create a forecast.
  >>> from intermittent_forecast.forecasters import TripleExponentialSmoothing
  >>> ts = [5, 6, 8, 9,
  ...       6, 8, 7,10,
  ...       8, 8, 9,12]
  >>> tes = TripleExponentialSmoothing().fit(
  ...     ts=ts,
  ...     period=4,
  ...     trend_type="additive",
  ...     seasonal_type="multiplicative",
  ...     alpha=0.3,
  ...     beta=0.1,
  ...     gamma=0.1,
  ... )
  >>> tes.forecast(start=0, end=11) # In-sample forecast
  array([ 5.13392857,  6.26839286,  8.44762143,  9.55915626,  5.30905245,
  6.75937741,  9.74957716, 10.22368613,  5.83750991,  8.01345859,
  10.43690141, 11.79762156])
  
  >>> # The out of sample forecasts is constructed from the final level,
  >>> # trend and seasonal component values.
  >>> tes.forecast(start=12, end=19)
  array([ 7.10671315,  8.4233535 , 10.77655469, 12.88922365,  7.69870712,
  9.11071018, 11.63835466, 13.89977022])
  
  >>> # Smoothing parameters can instead be optimised with a chosen
  >>> # error metric.
  >>> tes =TripleExponentialSmoothing().fit(
  ...     ts=ts,
  ...     period=4,
  ...     trend_type="additive",
  ...     seasonal_type="multiplicative",
  ...     optimisation_metric="MAE"
  ... )
  >>> tes.forecast(start=12, end=19)
  array([ 8.30291671,  9.27943232, 10.31641935, 13.4928251 ,  9.43460888,
  10.50254632, 11.6328387 , 15.16134003])
  
  >>> # Access a dict of the fitted values.
  >>> result = tes.get_fit_result()
  >>> result["alpha"], result["beta"], result["gamma"]
  (0.08364575434612503, 1.0, 0.47060129090469816)
```
<a id="intermittent_forecast.forecasters.triple_exponential_smoothing.TripleExponentialSmoothing.fit"></a>

#### `fit`

```python
def fit(
  ts: ArrayLike,
  period: int,
  trend_type: str = "additive",
  seasonal_type: str = "additive",
  alpha: float | None = None,
  beta: float | None = None,
  gamma: float | None = None,
  optimisation_metric: str | None = None,
) -> TripleExponentialSmoothing
```

Fit the model to the time-series.

**Arguments**:

- `ts` _ArrayLike_ - Time series to fit the model to. Must be
  1-dimensional and contain at least two non-zero values. If
  using multiplicative smoothing, the time series must be
  entirely positive.
- `period` _int_ - The period of the seasonal component.
- `trend_type` _str, optional_ - The type of trend smoothing to use.
  Options are "additive" or "multiplicative". Defaults to
  "additive". If using multiplicative smoothing, the time series
  must be entirely positive.
- `seasonal_type` _str, optional_ - The type of seasonal smoothing to
  use.Options are "additive" or "multiplicative". Defaults to
  "additive".If using multiplicative smoothing, the time series
  must be entirely positive.
- `alpha` _float, optional_ - Level smoothing factor in the range
  [0,1]. Values closer to 1 will favour recent demand. If not
  set, the value will be optimised.
- `beta` _float, optional_ - Trend smoothing factor in the range [0, 1].
  Values closer to 1 will favour recent demand. If not set, the
  value will be optimised.
- `gamma` _float, optional_ - Seasonal smoothing factor in the range
  [0,1]. Values closer to 1 will favour recent demand. If not
  set, the value will be optimised.
- `optimisation_metric` _str, optional_ - Metric to use when optimising
  for alpha and beta. Options are 'MAR', 'MAE', 'MSE', 'MSR',
  'PIS'. Defaults to 'MSE'. The selected metric is used when
  comparing the error between the time series and the fitted
  in-sample forecast.
  

**Returns**:

- `self` _TripleExponentialSmoothing_ - Fitted model instance.

<a id="intermittent_forecast.forecasters.triple_exponential_smoothing.TripleExponentialSmoothing.forecast"></a>

#### `forecast`

```python
def forecast(start: int, end: int) -> np.ndarray
```

Forecast the time series using the fitted parameters.

**Arguments**:

- `start` _int_ - Start index of the forecast (inclusive).
- `end` _int_ - End index of the forecast (inclusive).
  

**Returns**:

- `np.ndarray` - Forecasted values.

<a id="intermittent_forecast.forecasters.triple_exponential_smoothing.TripleExponentialSmoothing.get_fit_result"></a>

#### `get_fit_result`

```python
def get_fit_result() -> dict[str, Any]
```

Return the a dictionary of results if model has been fit.


