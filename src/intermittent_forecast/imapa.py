import numpy as np
from .adida import Adida
from .croston import croston


class Imapa:

    def __init__(self, ts):
        """
        Initialise IMAPA class
        
        Parameters
        ----------
        ts : array_like
            Input time series, 1-D list or array
        """
        self.ts = ts

    def agg(self, sizes, **kwargs):
        """
        Aggregate time series into different sized "buckets", with 
        an overlapping or non-overlapping window

        Parameters
        ----------
        sizes : list
            Aggregation sizes
        overlapping : bool
            Overlapping or non-overlapping window
        """        
        self.sizes = sizes
        self.adida_instances = []
        for s in sizes:
            adida = Adida(self.ts).agg(s, **kwargs)
            self.adida_instances.append(adida)
        return self

    def predict(self, **kwargs):
        """
        Create a forecast for each aggregated time series
                
        Parameters
        ----------
        fn : function
            Forecasting function, first parameter must be the input time series.
            Default is Croston's method.
        """
        for adida in self.adida_instances:
            adida.predict(**kwargs)
        return self
        
    def disagg(self, combine='mean', **kwargs):
        """
        Disaggregate each forecast to the original time scale. Combine the
        forecasts using the mean or median

        Parameters
        ----------
        combine : {'mean', 'median'}
            Method of combining the forecasts
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
        forecasts = []
        for adida in self.adida_instances:
            forecasts.append(adida.disagg(**kwargs))
        if combine == 'mean':
            return np.mean(forecasts, axis=0)
        elif combine == 'median':
            return np.median(forecasts, axis=0)