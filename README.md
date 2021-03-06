## Loan Default Prediction Machine Learning Project

This is an exploratory project for me to apply and compare different ML models and techniques, including:
+ Feature Engineering
  + One-hot encoding for categorical features
  + Normalization/standardization
  + Imputation
  + Feature expansion
  + Feature reduction:
    + Feature hashing
    + Feature selection
    + Principal component analysis
  + Feature discretization with Decision Trees or Random Forests
+ Machine Learning Models:
  + Logistic Regression
  + Decision Trees, Random Forest, Gradient-Boosted Decision Trees
  + K-Nearest-Neighbor
  + Support Vector Machines
  + Neural Networks

The data is from a Kaggle competition [Loan Default Prediction](https://www.kaggle.com/c/loan-default-prediction).

## Dependencies
Python 3, numpy, pandas, scikit-learn, matplotlib, xgboost, tensorflow, keras.

## Usage
Download train_v2.csv from https://www.kaggle.com/c/loan-default-prediction/data and put in the loan-default-prediction directory. Running the first Jupyter notebook, [LDP 01 - Data Preprocessing.ipynb](https://github.com/steggie3/loan-default-prediction/blob/master/LDP%2001%20-%20Data%20Preprocessing.ipynb) will give you a few processed CSV files, which are used in subsequent notebooks as the training data. The rest of the notebooks do not have strong dependencies.

