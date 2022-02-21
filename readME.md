



## Contents
1. [Task](###task)

2. [Data](###data)

3. [Requirements](###requirements)

4. [Files](###files)




### 1. Task

`Pet Adoption Classifier`
- This is a supervised model which trains an XGBoost Model to predict if a pet would be adopted or not.


### 2. Data

Contains 13 features and a labeled column (Yes or No) if a pet can be adopted. `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv`

### 3. Requirements

- gcsfs==2022.1.0
- numpy==1.19.5
- pandas==1.3.5
- scikit-learn==1.0.1
- xgboost==1.5.1
- Python==3.8.0
- matplotlib==3.5.1



### 4. Files
This directory contains 3 python 3 scripts
- `train.py`: Script to build and train a XGBoost classifier on the petFinder data (to run – python train.py)
- `predictor.py`: Script to test the saved XGBoost classifier on the whole data (to run – python predictor.py
- `test_predictor`: Unit Test script to automate testing of the model prediction function (to run – python predictor.py)


- - The directory also contains all the above script in 2 jupyter notebooks.


