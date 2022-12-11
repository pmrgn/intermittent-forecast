import numpy as np
from .adida import Adida
from .croston import croston


class Imapa():

    def __init__(self, ts):
        self.ts = ts

    def agg(self, sizes):
        ## TO DO REMOVE SIZES THAT WILL AGGREGATE TO <4
        self.sizes = sizes
        self.aggs = []
        for s in sizes:
            adida = Adida(self.ts).agg(s, overlapping=False)
            self.aggs.append(adida)
        return self

    def predict(self, **kwargs):
        for agg in self.aggs:
            agg.predict(**kwargs)
        return self
        
    def disagg(self, combine='mean', **kwargs):
        forecasts = []
        for agg in self.aggs:
            forecasts.append(agg.disagg(**kwargs))
        if combine == 'mean':
            return np.mean(forecasts, axis=0)
        elif combine == 'median':
            return np.median(forecasts, axis=0)
            


        
