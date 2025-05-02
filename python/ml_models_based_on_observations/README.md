# Simple time series prediction models based only on observational data
These models use only data from OBSTABLES and mainly use
TROAD and other sources of temperature to predict TROAD

##  Linear models

### simple_linear_model.py 
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
