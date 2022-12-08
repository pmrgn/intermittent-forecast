import numpy as np
from .croston import croston

class Adida():
    
    def __init__(self, ts):
        """
        Initialise Adida class
        
        Parameters
        ----------
        ts : array_like
            1-D list or array
        """
        self.ts = np.array(ts)
    
    def agg(self, size=1, overlapping=False):
        """
        Aggregate self.ts into "buckets", with an overlapping
        or non-overlapping window

        Parameters
        ----------
        size : int
            Size of aggregation window
        overlapping : bool
            Overlapping or non-overlapping window
        """
        self.size = size
        self.overlapping = overlapping
        if size == 1:
            self.aggregated = self.ts
            return self
        if self.overlapping:
            ts_agg = np.convolve(self.ts, np.ones(self.size), 'valid')  
        else:
            trim = len(self.ts) % self.size
            ts_trim = self.ts[trim:]
            ts_agg = ts_trim.reshape((-1, self.size)).sum(axis=1)
        self.aggregated = ts_agg
        return self

    def predict(self, fn=croston, *args, **kwargs):
        """
        Helper function, pass a forecasting function whose first parameter is
        the input time series. The aggregated time series will be passed to 
        this function followed by any arguments. 
                
        Parameters
        ----------
        fn : function
            Forecasting function
        """
        self.prediction = fn(self.aggregated, *args, **kwargs)
        return self
    
    def disagg(self, h=1, cycle=None, prediction=None,):
        """
        Disaggregate a prediction back to the original time scale

        Parameters
        ----------
        h : int
            Forecasting horizon, number of periods to forecast
        cycle : int, optional
            Number of periods in the seasonal cycle of the input time series. If not 
            defined, the disaggregation will be equal weighted
        prediction : int, optional
            Pass a single point prediction instead of using the predict method

        Returns
        -------
        forecast : ndarray
            1-D array of forecasted values
        """
        if not self.overlapping:
            p = np.repeat(self.prediction[:-1], self.size)
            offset = [np.nan] * (len(self.ts) % self.size)
            p = np.concatenate(
                (offset, p, self.prediction[-1:])
            )
        else:
            offset = [np.nan] * (self.size - 1)
            p = np.concatenate((offset, self.prediction))

        if cycle and cycle > 1:
            n = len(self.ts)
            trim = n % cycle
            s = self.ts[trim:].reshape(-1, cycle).sum(axis=0)
            frac = cycle / self.size
            s_perc = [s.sum() and (i * frac)/s.sum() for i in s]
            perc = s_perc * (n//cycle)  ## CHANGE TO VECTORISED MULTIPLICATION
            perc = np.concatenate(([np.nan]*trim,perc))   #
            f = np.array(s_perc) * p[-1]    # Out of sample forecast
            p = p[:-1] * perc               # In-sample forecast
        else:
            p = p / self.size
            p, f = p[:-1], p[-1]
        return np.concatenate((p,np.resize(f,h)))
