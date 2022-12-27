import numpy as np
from scipy.optimize import minimize
from ._error_metrics import mae, mse, msr, pis, mar


def _error(params, ts, method, metric):
    "Cost function used for optimisation of alpha and beta"
    alpha, beta = params
    dispatcher = {
        'mae': mae,
        'mse': mse,
        'mar': mar,
        'msr': msr,
        'pis': pis,
    }
    f = croston(ts, method=method, alpha=alpha, beta=beta,
                opt=False, metric=dispatcher[metric])
    if metric in ['msr', 'mar']:
        return dispatcher[metric](ts, f[:-1])
    return dispatcher[metric](ts, f[:-1])

def _auto_sel(ts):
    """
    Select the forecasting method using squared covariance of non-zero
    demand and mean demand interval
    """
    ts_nz = ts[ts != 0]
    p_mean = len(ts) / len(ts_nz)
    cv2 = (np.std(ts_nz, ddof=1) / np.mean(ts_nz))**2
    if cv2 <= 0.49 and p_mean <= 1.34:
        return 'cro'
    else:
        return 'sba'

def croston(ts, method='cro', alpha=None, beta=None, opt=True, metric='mar'):  
    """
    Perform smoothing on an intermittent time series, ts, and return
    a forecast array
    
    Parameters
    ----------
    ts : (N,) array_like
        1-D input time series
    method : {'cro', 'sba', 'sbj', 'tsb', 'auto'}
        Forecasting method: Croston, Syntetos-Boylan Approximation,
        Shale-Boylan-Johnston, Teunter-Syntetos-Babai. If 'auto', either
        Croston's method or SBA will be chosen based on CV^2 and mean
        demand interval.
    alpha : float
        Demand smoothing factor, `0 < alpha < 1`
    beta : float
        Interval smoothing factor, `0 < beta < 1`
    opt : boolean
        Optimise smoothing factors
    metric : {'mar', 'mae', 'mse', 'msr', 'pis'}
        Error metric to be used for optimisation of smoothing factors
        
    Returns
    -------
    forecast : (N+1,) ndarray
        1-D array of forecasted values
    """
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if len(ts[ts != 0]) < 2:
        raise ValueError('Time series needs at least two non-zero values')
    if alpha is None and opt == False:
        raise ValueError('Require a value for alpha, or set opt=True')
    if alpha:
        opt = False
        if not beta:
            beta = alpha
    if method == 'auto':
        method = _auto_sel(ts)
    if method == 'tsb':
        # Initialise demand array, z, and demand probability, p. The starting
        # value for z is the first non-zero demand value, starting value for p
        # is the inverse of the mean of all intervals
        n = len(ts)
        z = np.zeros(n)
        p = np.zeros(n)
        p_idx = np.flatnonzero(ts)
        p_diff = np.diff(p_idx, prepend=-1)
        z[0] = ts[p_idx[0]]
        p[0] = 1 / np.mean(p_diff)  # Probability of demand occurence

        # Optimise selection of alpha and beta if required
        if opt == True:
            init = [0.05,0.05]  # Initial guess for alpha, beta
            min_err = minimize(_error, init, 
                               args=(ts, method, metric), 
                               bounds=[(0,1), (0,1)])
            alpha, beta = min_err.x

        # Perform TSB
        for i in range(1,n):
            if ts[i] > 0:
                z[i] = alpha*ts[i] + (1-alpha)*z[i-1]
                p[i] = beta + (1-beta)*p[i-1]
            else:
                z[i] = z[i-1]
                p[i] = (1 - beta)*p[i-1]
        forecast = p * z
        forecast = np.insert(forecast, 0, np.nan)
        return forecast
    
    # CRO, SBA, SBJ:
    # Initialise arrays for demand, z, and period, p. Starting
    # demand is first non-zero demand value, starting period is
    # mean of all demand intervals
    nz = ts[ts != 0]
    p_idx = np.flatnonzero(ts)
    p_diff = np.diff(p_idx, prepend=-1)
    n = len(nz)
    z = np.zeros(n)
    p = np.zeros(n)
    z[0] = nz[0]
    p[0] = np.mean(p_diff)

    # Optimise selection of alpha and beta if required
    if opt == True:
        init = [0.05, 0.05]  # Initial guess for alpha, beta
        min_err = minimize(_error, init, 
                           args=(ts, method, metric), 
                           bounds=[(0,1), (0,1)])
        alpha, beta = min_err.x
        
    # Perform smoothing on demand and interval arrays
    for i in range(1, n):
        z[i] = alpha*nz[i] + (1-alpha)*z[i-1]
        p[i] = beta*p_diff[i] + (1-beta)*p[i-1]
    
    # Create forecast array, apply bias correction if required
    f = z / p
    if method == 'sba':
        f *= 1 - (beta/2)
    elif method == 'sbj':
        f *= 1 - (beta/(2-beta))

    # Return to original time scale by forward filling   
    z_idx = np.zeros(len(ts))
    z_idx[p_idx] = p_idx
    z_idx = np.maximum.accumulate(z_idx).astype('int')
    forecast = np.zeros(len(ts))
    forecast[p_idx] = f
    forecast = forecast[z_idx]

    # Starting forecast values up to and including first demand occurence
    # will be np.nan
    forecast[:p_idx[0]] = np.nan
    forecast = np.insert(forecast, 0, np.nan)
    return forecast
