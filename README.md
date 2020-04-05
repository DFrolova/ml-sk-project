# ml-sk-project

This is repository of team #8 (forecasters), ml course project.

The objective of this project is reproducing the paper "Meta-learning framework with applications to zero-shot time-series forecasting" https://arxiv.org/pdf/2002.02887.pdf

The paper is about pure deep learning approach to time series forecasting. Authors claim that their model is a new SOTA in time series forecasting.

Their architecture is called N-BEATS (https://arxiv.org/abs/1905.10437), in this project we used 1.3.0 version of N-BEATS https://pypi.org/project/nbeats-pytorch/#history.

The main tasks of the project are:

- Collect all datasets proposed by the publication and make them available in a shared drive (15%)
- provide a functioning and documented implementation of N-BEATS in pytorch (30%)
- to run baselines (20%)
- to conduct the experiments exactly as described in the publication (25%)
- for providing explanations on why N-BEATS works and when it does not (10%)

Concerning the first task, one can find scripts for loading time series from FRED (https://fred.stlouisfed.org/) in the corresponding folder. id_loader_final.ipynb is used to get id of time series from the site, series_loader_final.ipynb is used to load time series from FRED by their id. The main difference with existing APIs (e.g. could be found here https://fred.stlouisfed.org/docs/api/fred/) for loading time series from FRED is that we use Selenium for requesting id and time series, without Selenium it is not possible to download large amount of time series (~290k)

Concerning the second task, one could find implementation of N-BEATS in 
One can find N-BEATS training and predicting in the corresponding folder.

One can find statistical baselines for time series forecasting in the folder named baselines.


