# mle-training
# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Set up environment
 - Create conda environment with yml file
  ```
  conda env create -f env.yml
  ```

 - Activate the environment
  ```
  conda activate mle-dev
  ```
  
## To excute the script
python nonstandardcode.py 

## Install the package by installing build distribution
``` 
pip install dist/HousePricePrediction-0.3-py3-none-any.whl 
```

## To check the successful installation of the package , run the below command
``` 
%cd tests/functional_tests/
pytest test_installation.py
```
## To create a MLFlow server, run the below command
```
mlflow server --backend-store-uri mlruns/  --default-artifact-root mlruns/ --host 127.0.0.1 --port 5000
```

## To run the python script , run the below command
```
python src/HousePricePrediction/mlflow_run.py
```