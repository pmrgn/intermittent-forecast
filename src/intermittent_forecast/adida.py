import numpy as np
from .croston import croston


def _aggregate(ts, size, overlapping):
    # Aggregate 1-D ndarray using an overlapping or non-overlapping window
    if overlapping:
        return np.convolve(ts, np.ones(size), 'valid')
    else:
        trim = len(ts) % size
        ts_trim = ts[trim:]
        return ts_trim.reshape((-1, size)).sum(axis=1)


def _seasonal_cycle(ts, cycle):
    # Calculate the seasonal distribution for a time series
    pad = cycle - (len(ts) % cycle)
    ts = np.insert(ts.astype(float), 0, [np.nan]*pad)
    s = np.nanmean(ts.reshape(-1, cycle), axis=0)
    return np.array([s.sum() and i/s.sum() for i in s])


def _apply_cycle_perc(ts, cycle_perc):
    # Apply seasonal cycle percentages to a time series
    s = len(cycle_perc)
    pad = s - (len(ts) % s)
    ts = np.concatenate(([np.nan]*pad, ts))
    ts = (ts.reshape((-1, s)) * cycle_perc).flatten()
    return ts[pad:]


def adida(ts, size=1, overlapping=False, method='auto', opt=True,
          alpha=None, beta=None, metric='mar', h=1, cycle=None):
    """
    Aggregate-disaggregate Intermittent Demand Approach. Input time
    series is aggregated into "buckets" to reduce or remove 
    intermittency. One of the forecasting methods can then be applied
    to the aggregated series, followed by a seasonal or equal-weighted
    disaggregation.

    Parameters
    ----------
    ts : array_like
        Input time series, 1-D list or array
    size : int
        Size of aggregation window
    overlapping : bool
        Aggregate with an overlapping or non-overlapping window 
    method : {'auto', 'cro', 'sba', 'sbj', 'tsb'}
        Forecasting method: Croston, Syntetos-Boylan Approximation,
        Shale-Boylan-Johnston, Teunter-Syntetos-Babai. If 'auto', either
        Croston's method or SBA will be chosen based on CV^2 and mean
        demand interval.
    alpha : float
        Demand smoothing factor, `0 < alpha < 1`
    beta : float
        Interval smoothing factor, `0 < beta < 1`
    opt : boolean
        Optimise smoothing factors. If a value for alpha is passed,
        then optimisation will not occur.
    metric : {'mar', 'mae', 'mse', 'msr', 'pis'}
        Error metric to be used for optimisation of smoothing factors
    h : int
        Forecasting horizon, number of periods to forecast
    cycle : int, optional
        For a seasonal disaggregation, enter the number of periods in the 
        seasonal cycle of the input time series. If not defined, the 
        disaggregation will be equal weighted.

    Returns
    -------
    forecast : ndarray
        1-D array of forecasted values        
    """
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if size == 1:
        ts_agg = ts
    else:
        ts_agg = _aggregate(ts, size, overlapping)
    fc = croston(ts=ts_agg, method=method, alpha=alpha,
                 beta=beta, opt=opt, metric=metric)
    fc_in = fc[:-1]   # In-sample forecast
    fc_out = fc[-1]   # Out of sample forecast
    if not overlapping:
        nan_pad = [np.nan] * (len(ts) % size)
        fc_in = fc_in.repeat(size)
        fc_in = np.concatenate((nan_pad, fc_in))
    else:
        nan_pad = [np.nan] * (size - 1)
        fc_in = np.concatenate((nan_pad, fc_in))
    if cycle and cycle > 1:
        cycle_perc = _seasonal_cycle(ts, cycle)
        cycle_perc = cycle_perc * (cycle / size)
        fc_in = _apply_cycle_perc(fc_in, cycle_perc)
        fc_out = fc_out * cycle_perc
    else:
        fc_in /= size
        fc_out /= size
    return np.concatenate((fc_in, np.resize(fc_out, h)))
