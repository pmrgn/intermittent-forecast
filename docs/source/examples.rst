Examples
============

Croston's Method
****************

To perform Croston's method on a time series, import the croston module. The module contains
adaptations such as Syntetos-Boylan Approximation (sba), Shale-Boylan-Johnston (sbj), 
Teunter-Syntetos-Babai (tsb) as well as an auto-selection method (auto).

Smoothing factors, alpha and beta, can be set manually.

::

    >>> from intermittent_forecast import croston

    >>> ts = np.random.randint(0,2,100)
    >>> croston(ts, method='sba', alpha=0.1, beta=0.05)
    array([nan, 0.6825, 0.0.6928934, 0.6928934, ...,])

An optimised fit can also be performed which will select values for alpha and beta 
such that the chosen metric is minimised.

.. code-block:: python

    >>> croston(ts, method='sba', opt=True, metric='mse')
    array([nan, 0.6999993 , 0.69999972, 0.69999972, ...,])

ADIDA
*****

Aggregate-Disaggregate Intermittent Demand Approach. 

The input time series is aggregated into "buckets" to reduce or remove 
intermittency. One of the forecasting methods from the Croston module
can then be applied to the aggregated series, followed by a seasonal or e
qual-weighted disaggregation.

.. code-block:: python

    >>> from intermittent_forecast import adida

    >>> ts = np.random.randint(0,2,100)
    >>> adida(ts, size=3, overlapping=True, method='sba', alpha=0.1)
    array([nan, nan, nan, nan, 0.2462963, 0.27708333, ...])


IMAPA
*****

Intermittent Multiple Aggregation Prediction Algorithm. 

The input time series is aggregated using multiple aggregation sizes. A forecast
for each aggregation size is calculated, then either the mean or median is taken to
produce a single forecast series.

.. code-block:: python

    >>> from intermittent_forecast import imapa

    >>> ts = np.random.randint(0,2,100)
    >>> imapa(ts, sizes=[1,2,3], combine='mean', overlapping=True)
    array([nan, nan, nan, 0.5713449, 0.59967533, ...])  