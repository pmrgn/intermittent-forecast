import numpy as np


def mae(ts, forecast):
    "Return mean absolute error of two numpy arrays"
    return np.mean(np.abs(ts - forecast))


def mse(ts, forecast):
    "Return mean squared error of two numpy arrays"
    return np.mean((ts - forecast)**2)


def msr(ts, forecast):
    "Return mean squared rate of two numpy arrays"
    n = len(ts)
    d_rate = np.cumsum(ts) / np.arange(1,n+1)  # Demand rate
    bfill = int(np.ceil(0.3*n))  # Backfill first 30% of values
    d_rate[:bfill] = d_rate[bfill]
    idx = np.argmax(np.isfinite(forecast))
    return mse(d_rate[idx:], forecast[idx:])


def pis(ts, forecast):
    "Return absolute periods in stock of two numpy arrays"
    cfe = np.cumsum(ts - forecast)
    pis = -np.cumsum(cfe)
    return np.abs(pis[-1])
