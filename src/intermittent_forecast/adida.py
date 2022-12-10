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
        Pass a forecasting function whose first parameter is
        the input time series. The aggregated time series will be passed to 
        this function followed by any arguments. 
                
        Parameters
        ----------
        fn : function
            Forecasting function
        """
        self.prediction = fn(self.aggregated, *args, **kwargs)
        return self
    
    def disagg(self, h=1, cycle=None):
        """
        Disaggregate a prediction back to the original time scale

        Parameters
        ----------
        h : int
            Forecasting horizon, number of periods to forecast
        cycle : int, optional
            Number of periods in the seasonal cycle of the input time series. If not 
            defined, the disaggregation will be equal weighted

        Returns
        -------
        forecast : ndarray
            1-D array of forecasted values
        """
        f_in = self.prediction[:-1]   # In-sample forecast
        f_out = self.prediction[-1]   # Out of sample forecast
        if not self.overlapping:
            f_in = f_in.repeat(self.size)
            offset = [np.nan] * (len(self.ts) % self.size)
            f_in = np.concatenate((offset, f_in))
        else:
            offset = [np.nan] * (self.size - 1)
            f_in = np.concatenate((offset, f_in))
        if cycle and cycle > 1:
            # Calculate the seasonal percentage for each step in the cycle 
            n = len(self.ts)
            trim = n % cycle
            s = self.ts[trim:].reshape(-1, cycle).sum(axis=0)
            frac = cycle / self.size
            s_perc = np.array([s.sum() and (i * frac)/s.sum() for i in s]) 
            # Apply seasonal percentage to f_in. First pad with nan to be able to
            # reshape for a vectorised multiplication, then remove padding 
            f_in = np.concatenate(([np.nan]*trim, f_in))
            f_in = (f_in.reshape((-1,len(s_perc))) * s_perc).flatten()
            f_in = f_in[trim:]
            f_out = f_out * s_perc
        else:
            f_in /= self.size
            f_out /= self.size
        return np.concatenate((f_in,np.resize(f_out,h)))
