import numpy as np


def mae(ts, forecast):
    "Return mean absolute error of two numpy arrays"
    e = ts - forecast
    e = e[~np.isnan(e)]
    return np.mean(np.abs(e))


def mse(ts, forecast):
    "Return mean squared error of two numpy arrays"
    e = ts - forecast
    e = e[~np.isnan(e)]
    return np.mean(e**2)


def _demand_rate(ts):
    "Return demand rate of a time series"
    n = len(ts)
    d_rate = np.cumsum(ts) / np.arange(1,n+1)  # Demand rate
    bfill = int(np.ceil(0.3*n))  # Backfill first 30% of values
    d_rate[:bfill] = d_rate[bfill - 1]
    return d_rate


def msr(ts, forecast):
    "Return mean squared rate of two numpy arrays"
    d_rate = _demand_rate(ts)
    return mse(d_rate, forecast)


def mar(ts, forecast):
    "Return mean absolute rate of two numpy arrays"
    d_rate = _demand_rate(ts)
    return mae(d_rate, forecast)


def pis(ts, forecast):
    "Return absolute periods in stock of two numpy arrays"
    e = ts - forecast
    e = e[~np.isnan(e)]
    cfe = np.cumsum(e)
    pis = -np.cumsum(cfe)
    return np.abs(pis[-1])


