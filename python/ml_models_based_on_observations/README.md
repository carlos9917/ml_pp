# Simple time series prediction models based only on observational data
These models use only data from OBSTABLES and mainly use
TROAD and other sources of temperature to predict TROAD


## simplest linear model
This is the simplest model that uses an 80/20 split 
in training and test. 
The model assumes there is enough data for a given station.

```
python simple_linear_model.py --year 2023 --station 630200

```


If it fails use the option `--analyze-stations` to dump
a list of stations with the most data available.
```
python simple_linear_model.py --year 2023 --station 100001 --analyze-stations

```


## LSTM model

Run adding a train to test ratio using the option `train_ratio`
This will predict 40 hours

python pytorch_lstm_road_temp.py --train_ratio 0.995 --lookback 24
