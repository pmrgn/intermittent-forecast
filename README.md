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




