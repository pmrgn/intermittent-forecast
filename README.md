# Forecasting Intermittent Time Series 

This package contains tools for forecasting intermittent time series. The outputs will be similar to those obtained from using the R package, [tsintermittent](https://cran.r-project.org/web/packages/tsintermittent/index.html). 

## Installation

Clone the repo

	git clone git@github.com:pmrgn/intermittent-forecast.git
	
Navigate to the package folder and install using `pip`

	pip install .

### Croston's Method

Croston's method and several adaptations are available in the `croston` function. 

	from intermittent_forecast import croston

Pass a time series of length N to the `croston`, which will return a forecast of length N+1. Smoothing factors are set with the `alpha` and `beta` paramters. The variations of Croston's method can selected by passing one of the following to the `method` parameter.
- 'cro' : Croston
- 'sba' : Syntetos-Boylan Approximation
- 'sbj' : Shale-Boylan-Johnston
- 'tsb' : Teunter-Syntetos-Babai
<!-- End of List -->

	ts = np.array([0,1,0,2,1])
	forecast = croston(ts, method='sba', alpha=0.2, beta=0.1)
	
Smoothing parameters (alpha, beta) can also be optimised by passing `opt=True` and using one of the following cost functions. 
- 'mae' : Mean Absolute Error
- 'mse' : Mean Squared Error
- 'msr' : Mean Squared Rate
- 'pis' : Periods in Stock
<!-- End of List -->

	forecast = croston(ts, method='tsb', opt=True, metric='msr')

### ADIDA Class

The Aggregate Disaggregate Intermittent Demand Approach (ADIDA) is available through the Adida class

	from intermittent_forecast import Adida

Create an instance of the class by passing a time series

	ts = np.arange(20)
	adida = Adida(ts)

Aggregate the series into "buckets" by calling the `agg` method with the aggregation size and either an overlapping or non-overlapping window.

	adida.agg(size=4, overlapping=False)

A forecast can be calculated using the `predict` method and passing a forecasting function whose first parameter is the input time series. For example, using Croston's method within this package.

	from intermittent_forecast import croston

	adida.predict(croston, method='sba', opt=True, metric='msr')

The forecast can then be disaggregated back to the original time scale by calling the `disagg` method, which will return the forecast array h-steps into the future. To perform a seasonal disaggregation, pass a value for the seasonal cycle. 

	forecast = adida.disagg(h=10, cycle=4)

### IMAPA Class

The Multiple Aggregation Prediction Algorithm (MAPA) approach is found in the `Imapa` class. It will create multiple instances of the Adida class and combine the end results, using either the mean or median. All parameters in the Adida class methods can be passed to the Imapa methods. Repeating the example from the Adida class but with multiple aggregation sizes.

    from intermittent_forecast import Imapa, croston

    ts = np.arange(20)
    sizes = np.arange(1,5)

    imapa = (
        Imapa(ts)
        .agg(sizes, overlapping=False)
        .predict(croston, method='sba', opt=True, metric='msr')
        .disagg(combine='mean', h=10, cycle=4)
    )
