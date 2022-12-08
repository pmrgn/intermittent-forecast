import numpy as np
from .adida import Adida
from .croston import croston


class Imapa():

    def __init__(self, ts):
        self.adida = Adida(ts)

    def agg(self, sizes):
        ## TO DO REMOVE SIZES THAT WILL AGGREGATE TO <4
        self.sizes = sizes
        self.aggregated = []
        for s in sizes:
            ts_agg = self.adida.agg(s, overlapping=False).aggregated
            self.aggregated.append(ts_agg)
        return self

    def _model_sel(self, ts):
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

    def predict(self, alpha=None, beta=None, opt=True, metric='mar',
                combine='mean', h=1, cycle=None):
        predictions = []
        if alpha:
            opt = False
            if not beta:
                beta = alpha
        for ts in self.aggregated:
            method = self._model_sel(ts)
            print(method)
            pred = croston(ts, method=method, alpha=alpha, beta=beta,
                           opt=opt, metric=metric)
            predictions.append(pred)
        for i in range(len(self.sizes)):
            if self.sizes[i] != 1:
                s = self.sizes[i]
                predictions[i] = predictions[i].repeat(s)[:-s+1] / s
        if combine == 'mean':
            forecast = np.mean(predictions, axis=0)
        else:
            forecast = np.median(predictions, axis=0)
        return forecast
        # self.adida.size = 1
        # f = self.adida.disagg(h=h, cycle=cycle, prediction=pred)
        # return f


        
