import numpy as np
from intermittent_forecast import Imapa, croston, error_metrics, Adida

ts = np.arange(1,21)
adida = (
    Adida(ts).agg(size=1, overlapping=False)
    .predict(method='cro', alpha=1)
    .disagg(h=1)
)
print(adida)

imapa = Imapa(ts).agg(sizes=[1,2,3,4,5])
for agg in imapa.aggs:
    print(agg.aggregated)
imapa = imapa.predict(method='cro', alpha=1).disagg(h=1, combine='median')
print(imapa)