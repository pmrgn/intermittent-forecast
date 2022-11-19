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
        if overlapping:
            ts_agg = np.convolve(ts, np.ones(self.size), 'valid')  
        else:
            trim = len(self.ts) % self.size
            ts_trim = self.ts[trim:]
            ts_agg = ts_trim.reshape((-1, self.size)).sum(axis=1)
        self.aggregated = np.trim_zeros(ts_agg, 'f')
        return self

    def predict(self, fn=croston, *args, **kwargs):
        """
        Helper function, pass a forecasting function whose first parameter is
        the input time series. The aggregated time series will be passed to 
        this function followed by any arguments. If the forecasting function returns 
        an array, the final value will be used as the prediction.
                
        Parameters
        ----------
        fn : function
            Forecasting function
        """
        self.pred = fn(self.aggregated, *args, **kwargs)
        if len(self.pred) > 1:
            self.pred = self.pred[-1]
        return self
    
    def disagg(self, prediction=None, h=1, cycle=None):
        """
        Disaggregate a prediction back to the original time scale

        Parameters
        ----------
        prediction : int, optional
            Pass a single point prediction if the predict() method hasn't
            been used to generate a forecast
        h : int
            Forecasting horizon, number of periods to forecast
        cycle : int, optional
            Number of periods in the seasonal cycle of the input time series. If not 
            defined, the disaggregation will be equal weighted

        Returns
        -------
        forecast : ndarray
            1-D array of forecasted values of size (h,)
        """
        if prediction:
            self.pred = prediction
        if cycle:
            trim = len(self.ts) % cycle
            s = self.ts[trim:].reshape(-1, cycle).sum(axis=0)
            s_perc = [s.sum() and i/s.sum() for i in s]  # Short-circuit for zero division
            pred = self.pred * (cycle/self.size)
            return np.resize([i * pred for i in s_perc],h)
        else:
            return np.array([self.pred/self.size] * h) 
