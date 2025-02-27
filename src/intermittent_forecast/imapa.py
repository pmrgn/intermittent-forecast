import numpy as np

from intermittent_forecast.adida import adida


def imapa(ts, sizes=[1], combine='mean', overlapping=False, method='auto',
          opt=True, alpha=None, beta=None, metric='mar', h=1, cycle=None):
    """
    Intermittent Multiple Aggregation Prediction Algorithm. For each aggregation
    level, ADIDA will be performed. The resulting forecasts will then be 
    combined.    

    Parameters
    ----------
    ts : array_like
        Input time series, 1-D list or array
    sizes : list
        Aggregation sizes to use, list of integers.
    combine : {'mean', 'median'}
        Combine the forecasts using either the mean or median
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
        seasonal cycle of the input time series. If not If not defined, the 
        disaggregation will be equal weighted.

    Returns
    -------
    forecast : ndarray
        1-D array of forecasted values        
    """
    forecasts = []
    for size in sizes:
        f = adida(ts, size=size, overlapping=overlapping, method=method, 
                  opt=opt, alpha=alpha, beta=beta,metric=metric, h=h, 
                  cycle=cycle)
        forecasts.append(f)
    forecasts = np.array(forecasts)
    if len(sizes) == 1:
        return forecasts[0]
    if combine == 'mean':
        return np.mean(forecasts, axis=0)
    else:
        return np.median(forecasts, axis=0)

        