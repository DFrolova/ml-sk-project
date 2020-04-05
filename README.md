# ml-sk-project

This is repository of team #8 (forecasters), containing scripts for "Machine Learning and Applications" course project.

The objective of this project is reproducing the paper "Meta-learning framework with applications to zero-shot time-series forecasting" https://arxiv.org/pdf/2002.02887.pdf.

The paper is about pure deep learning approach to time series forecasting. Authors claim that their model is a new SOTA in time series forecasting.

Their architecture is called N-BEATS (https://arxiv.org/abs/1905.10437), in this project we used 1.3.0 version of N-BEATS https://pypi.org/project/nbeats-pytorch/#history.

The main tasks of the project are:

- Collect all datasets proposed by the publication and make them available in a shared drive (15%)
- provide a functioning and documented implementation of N-BEATS in pytorch (30%)
- to run baselines (20%)
- to conduct the experiments exactly as described in the publication (25%)
- for providing explanations on why N-BEATS works and when it does not (10%)

1)  Concerning the first task, one can find scripts for loading time series from FRED (https://fred.stlouisfed.org/) in the corresponding folder.
- **id_loader_final.ipynb** is used to get id of time series from the site
- **series_loader_final.ipynb** is used to load time series from FRED by their id

The main difference with existing APIs (e.g. could be found here https://fred.stlouisfed.org/docs/api/fred/) for loading time series from FRED is that we use Selenium for requesting id and time series, without Selenium it is not possible to download large amount of time series (~180k). Scripts are written with instructions. In order to collect other datasets one do not need to use any scripts. All collected datasets one can find here https://drive.google.com/drive/folders/1U8He5OXbLHBjDzm8AX8dDme_2ygnXhmy .

2)  Concerning running baselines, one could find script written in R **Baselines.r**.
...
Maxim's description
...

3)  Concerning the other tasks, one can find N-BEATS training and predicting scripts in the corresponding folder.

3.1)  Notebook **train_nbeats_1.ipynb** is for training ensembles of N-BEATS. There are different data frequences (data_types) for both training datasets (M4 and FRED).
Parameters to set:
- data_type: ('yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly') for M4, ('yearly', 'quarterly', 'monthly') for fred - frequency to train on
- dataset: 'm4' or 'fred' - dataset to train on

Notebook saves trained models to a FOLDER (may be changed) into a file CHECKPOINT_NAME.

3.2)  In order to predict time series one need to use **ML_project_final.ipynb**.
There are parameters to change in the main block of notebook:
data_type: frequency of data which is needed to predict. May be yearly, quarterly, monthly, daily, weekly, hourly, other depending on the dataset
- train_dataset: 'm4' or 'fred' - dataset on which models were trained
- test_dataset: dataset on which we want to get predictions. Can be m4, m3, fred, electricity, traffic, tourism
This code uses files CHECKPOINT_NAME in FOLDER as path to trained models - should be changed if you use another path.
Also, it saves predictions to file PREDS_FILENAME.
After execution of this cell metric on an ensemble of models is printed, and a picture with visualization of predictions of first 9 time series is saved into file 'n_beats_111.png'. There blue line corresponds to historical data of time series, green line - true values, red line - predictions.

3.3) There is additional notebook **Preprocess_FRED.ipynb** for preprocessing FRED data. It is needed because some time series of FRED contains None, some time series are constants, some time series are too short, etc.

In the figure below one can see how well Meta Learning approach based on N-BEATS model can predict time series. Green are real time series, red is predicted by N-BEATS. Model was trained on M4, here predictions for Traffic dataset.


In the table below there are averaged results of model performance on different datasets.


| model / test dataset | M4 | FRED | 
|----------------------|----|------|
| SNa\"{\i}ve | 15.20 | 22.26 |
| Theta 12.70 | 22.27 |
| ARIMA | 13.20 | 22.02 |
| N-BEATS(M4) | - | 20.68 |
| N-BEATS(FRED) | 13.56 | - |
