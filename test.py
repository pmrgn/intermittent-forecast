import numpy as np
from intermittent_forecast import Imapa, croston, error_metrics, Adida

ts = [1,2,3,4] * 4
adida = (
    Adida(ts).agg(size=4, overlapping=False)
    .predict(croston, method='cro', alpha=1)
    .disagg(h=4, cycle=4)
)
print(adida)
exp = np.append([np.nan]*4,ts)
print(exp)