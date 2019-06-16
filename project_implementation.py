
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from math import sqrt

warnings.filterwarnings('ignore')  # for suppressing unwanted warnings

dataset = pd.read_csv('forestfires.csv') # importing the dataset

# Preprocessing the dataset so that for particular month, particular month number is assigned
labels = dataset.pop('area')
for j in range(len(dataset['month'])):
    
    if dataset['month'][j] == 'jan':
        dataset['month'][j] = 1
        
    elif dataset['month'][j] == 'feb':
        dataset['month'][j] = 2
        
    elif dataset['month'][j] == 'mar':
        dataset['month'][j] = 3
        
    elif dataset['month'][j] == 'apr':
        dataset['month'][j] = 4
        
    elif dataset['month'][j] == 'may':
        dataset['month'][j] = 5
        
    elif dataset['month'][j] == 'jun':
        dataset['month'][j] = 6
        
    elif dataset['month'][j] == 'jul':
        dataset['month'][j] = 7
        
    elif dataset['month'][j] == 'aug':
        dataset['month'][j] = 8
        
    elif dataset['month'][j] == 'sep':
        dataset['month'][j] = 9
        
    elif dataset['month'][j] == 'oct':
        dataset['month'][j] = 10
        
    elif dataset['month'][j] == 'nov':
        dataset['month'][j] = 11
        
    elif dataset['month'][j] == 'dec':
        dataset['month'][j] = 12


# Preprocessing the dataset so that for the particular week-day, particular weak-day number is assigned
for j in range(len(dataset['day'])):
    
    if dataset['day'][j] == 'mon':
        dataset['day'][j] = 1
        
    elif dataset['day'][j] == 'tue':
        dataset['day'][j] = 2
        
    elif dataset['day'][j] == 'wed':
        dataset['day'][j] = 3
        
    elif dataset['day'][j] == 'thu':
        dataset['day'][j] = 4
        
    elif dataset['day'][j] == 'fri':
        dataset['day'][j] = 5
        
    elif dataset['day'][j] == 'sat':
        dataset['day'][j] = 6
        
    elif dataset['day'][j] == 'sun':
        dataset['day'][j] = 7
        
# Standardizing the dataset 
'''Standardization refers to shifting the distribution of each attribute to have a mean of zero 
and a standard deviation of one (unit variance).'''

scaled_features = StandardScaler().fit_transform(dataset.values)

# Creating back the dataframe of the scaled data
dataset = pd.DataFrame(scaled_features, index=dataset.index, columns=dataset.columns)

# Partitioning the dataset into test and train (test = (25% of total data) and train = (75% of total data) )
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.25, random_state = 4)

# Importing all the required Regressors used in the paper

clf_nb = BayesianRidge(compute_score=True) # Naive-Bayes Ridge Regressor
clf_mr = LinearRegression() # Multiple-Regression
clf_dt = DecisionTreeRegressor(max_depth=2,random_state=0) # Decision Tree Regressor
clf_rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=10) # Random Forest Regressor
clf_nn = MLPRegressor() # Neural Networks Regressor
clf_svm = SVR(C=1.0, epsilon=0.2) # SVM Regressor



# In[3]:


# for STFWI : Spatial, temporal, and Forest Waether Index features are taken
train_features_one = X_train[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI']].copy()
test_features_one = X_test[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI']].copy()

# Naive Bayes Regressor

# Training using Naive Bayes Regressor
clf_nb.fit(train_features_one, y_train)
# Testing using Naive Bayes Regressor
predictions_nb_one = clf_nb.predict(test_features_one)

# RootMeanSquared Error Calculation for Naive Bayes Regressor
print("\n\nFor STFWI features using naive_bayes_ridge")
meanSquaredError_nb_one = mean_squared_error(y_test, predictions_nb_one)
#print("MSE:", meanSquaredError_nb_one)
rootMeanSquaredError_nb_one = sqrt(meanSquaredError_nb_one)
print("RMSE:", rootMeanSquaredError_nb_one)

# MeanAbsolute Error Calculation for Naive Bayes Regressor
absolute_error_nb_one = mean_absolute_error(y_test, predictions_nb_one)
print("Absolute error is:", absolute_error_nb_one)



# Linear Regression

# Training using Linear Regression
clf_mr.fit(train_features_one, y_train)
# Testing using Linear Regression
predictions_mr_one = clf_mr.predict(test_features_one)

# RootMeanSquared Error Calculation for Linear Regression
print("\n\nFor STFWI features using Linear Regression")
meanSquaredError_mr_one = mean_squared_error(y_test, predictions_mr_one)
#print("MSE:", meanSquaredError_mr_one)
rootMeanSquaredError_mr_one = sqrt(meanSquaredError_mr_one)
print("RMSE:", rootMeanSquaredError_mr_one)

# MeanAbsolute Error Calculation for Linear Regression
absolute_error_mr_one = mean_absolute_error(y_test, predictions_mr_one)
print("Absolute error is:", absolute_error_mr_one)



# Decision Trees Regressor

# Training using Decision Trees Regressor
clf_dt.fit(train_features_one, y_train)
# Testing using Decision Trees Regressor
predictions_dt_one = clf_dt.predict(test_features_one)

# RootMeanSquared Error Calculation for Decision Trees Regressor
print("\n\nFor STFWI features using Decision Trees Regressor")
meanSquaredError_dt_one = mean_squared_error(y_test, predictions_dt_one)
#print("MSE:", meanSquaredError_dt_one)
rootMeanSquaredError_dt_one = sqrt(meanSquaredError_dt_one)
print("RMSE:", rootMeanSquaredError_dt_one)

# MeanAbsolute Error Calculation for Decision Trees Regressor
absolute_error_dt_one = mean_absolute_error(y_test, predictions_dt_one)
print("Absolute error is:", absolute_error_dt_one)



# Random Forest Regressor

# Training using Random Forest Regressor
clf_rf.fit(train_features_one, y_train)
# Testing using Random Forest Regressor
predictions_rf_one = clf_rf.predict(test_features_one)

# RootMeanSquared Error Calculation for Random Forest Regressor
print("\n\nFor STFWI features using Random Forest Regressor")
meanSquaredError_rf_one = mean_squared_error(y_test, predictions_rf_one)
#print("MSE:", meanSquaredError_rf_one)
rootMeanSquaredError_rf_one = sqrt(meanSquaredError_rf_one)
print("RMSE:", rootMeanSquaredError_rf_one)

# MeanAbsolute Error Calculation for Random Forest Regressor
absolute_error_rf_one = mean_absolute_error(y_test, predictions_rf_one)
print("Absolute error is:", absolute_error_rf_one)



# MLP Regressor

# Training using MLP Regressor
clf_nn.fit(train_features_one, y_train)
# Testing using MLP Regressor
predictions_nn_one = clf_nn.predict(test_features_one)

# RootMeanSquared Error Calculation for MLP Regressor
print("\n\nFor STFWI features using MLP Regressor")
meanSquaredError_nn_one = mean_squared_error(y_test, predictions_nn_one)
#print("MSE:", meanSquaredError_nn_one)
rootMeanSquaredError_nn_one = sqrt(meanSquaredError_nn_one)
print("RMSE:", rootMeanSquaredError_nn_one)

# MeanAbsolute Error Calculation for MLP Regressor
absolute_error_nn_one = mean_absolute_error(y_test, predictions_nn_one)
print("Absolute error is:", absolute_error_nn_one)



# SVM Regressor

# Training using SVM Regressor
clf_svm.fit(train_features_one, y_train)
# Testing using SVM Regressor
predictions_svm_one = clf_svm.predict(test_features_one)

# RootMeanSquared Error Calculation for SVM Regressor
print("\n\nFor STFWI features using SVM Regressor")
meanSquaredError_svm_one = mean_squared_error(y_test, predictions_svm_one)
#print("MSE:", meanSquaredError_svm_one)
rootMeanSquaredError_svm_one = sqrt(meanSquaredError_svm_one)
print("RMSE:", rootMeanSquaredError_svm_one)

# MeanAbsolute Error Calculation for SVM Regressor
absolute_error_svm_one = mean_absolute_error(y_test, predictions_svm_one)
print("Absolute error is:", absolute_error_svm_one)


# In[4]:


# for STM : Spatial, temporal, and Meterological features are taken
train_features_two = X_train[['X', 'Y', 'month', 'day', 'temp', 'RH', 'wind', 'rain']].copy()
test_features_two = X_test[['X', 'Y', 'month', 'day', 'temp', 'RH', 'wind', 'rain']].copy()


# Naive Bayes Regressor

# Training using Naive Bayes Regressor
clf_nb.fit(train_features_two, y_train)
# Testing using Naive Bayes Regressor
predictions_nb_two = clf_nb.predict(test_features_two)

# RootMeanSquared Error Calculation for Naive Bayes Regressor
print("\n\nFor STM features using naive_bayes_ridge")
meanSquaredError_nb_two = mean_squared_error(y_test, predictions_nb_two)
#print("MSE:", meanSquaredError_nb_two)
rootMeanSquaredError_nb_two = sqrt(meanSquaredError_nb_two)
print("RMSE:", rootMeanSquaredError_nb_two)

# MeanAbsolute Error Calculation for Naive Bayes Regressor
absolute_error_nb_two = mean_absolute_error(y_test, predictions_nb_two)
print("Absolute error is:", absolute_error_nb_two)



# Linear Regression

# Training using Linear Regression
clf_mr.fit(train_features_two, y_train)
# Testing using Linear Regression
predictions_mr_two = clf_mr.predict(test_features_two)

# RootMeanSquared Error Calculation for Linear Regression
print("\n\nFor STM features using Linear Regression")
meanSquaredError_mr_two = mean_squared_error(y_test, predictions_mr_two)
#print("MSE:", meanSquaredError_mr_two)
rootMeanSquaredError_mr_two = sqrt(meanSquaredError_mr_two)
print("RMSE:", rootMeanSquaredError_mr_two)

# MeanAbsolute Error Calculation for Linear Regression
absolute_error_mr_two = mean_absolute_error(y_test, predictions_mr_two)
print("Absolute error is:", absolute_error_mr_two)



# Decision Trees Regressor

# Training using Decision Trees Regressor
clf_dt.fit(train_features_two, y_train)
# Testing using Decision Trees Regressor
predictions_dt_two = clf_dt.predict(test_features_two)

# RootMeanSquared Error Calculation for Decision Trees Regressor
print("\n\nFor STM features using Decision Trees Regressor")
meanSquaredError_dt_two = mean_squared_error(y_test, predictions_dt_two)
#print("MSE:", meanSquaredError_dt_two)
rootMeanSquaredError_dt_two = sqrt(meanSquaredError_dt_two)
print("RMSE:", rootMeanSquaredError_dt_two)

# MeanAbsolute Error Calculation for Decision Trees Regressor
absolute_error_dt_two = mean_absolute_error(y_test, predictions_dt_two)
print("Absolute error is:", absolute_error_dt_two)



# Random Forest Regressor

# Training using Random Forest Regressor
clf_rf.fit(train_features_two, y_train)
# Testing using Random Forest Regressor
predictions_rf_two = clf_rf.predict(test_features_two)

# RootMeanSquared Error Calculation for Random Forest Regressor
print("\n\nFor STM features using Random Forest Regressor")
meanSquaredError_rf_two = mean_squared_error(y_test, predictions_rf_two)
#print("MSE:", meanSquaredError_rf_two)
rootMeanSquaredError_rf_two = sqrt(meanSquaredError_rf_two)
print("RMSE:", rootMeanSquaredError_rf_two)

# MeanAbsolute Error Calculation for Random Forest Regressor
absolute_error_rf_two = mean_absolute_error(y_test, predictions_rf_two)
print("Absolute error is:", absolute_error_rf_two)



# MLP Regressor

# Training using MLP Regressor
clf_nn.fit(train_features_two, y_train)
# Testing using MLP Regressor
predictions_nn_two = clf_nn.predict(test_features_two)

# RootMeanSquared Error Calculation for MLP Regressor
print("\n\nFor STM features using MLP Regressor")
meanSquaredError_nn_two = mean_squared_error(y_test, predictions_nn_two)
#print("MSE:", meanSquaredError_nn_two)
rootMeanSquaredError_nn_two = sqrt(meanSquaredError_nn_two)
print("RMSE:", rootMeanSquaredError_nn_two)

# MeanAbsolute Error Calculation for MLP Regressor
absolute_error_nn_two = mean_absolute_error(y_test, predictions_nn_two)
print("Absolute error is:", absolute_error_nn_two)



# SVM Regressor

# Training using SVM Regressor
clf_svm.fit(train_features_two, y_train)
# Testing using SVM Regressor
predictions_svm_two = clf_svm.predict(test_features_two)

# RootMeanSquared Error Calculation for SVM Regressor
print("\n\nFor STM features using SVM Regressor")
meanSquaredError_svm_two = mean_squared_error(y_test, predictions_svm_two)
#print("MSE:", meanSquaredError_svm_two)
rootMeanSquaredError_svm_two = sqrt(meanSquaredError_svm_two)
print("RMSE:", rootMeanSquaredError_svm_two)

# MeanAbsolute Error Calculation for SVM Regressor
absolute_error_svm_two = mean_absolute_error(y_test, predictions_svm_two)
print("Absolute error is:", absolute_error_svm_two)


# In[5]:


# for FWI : Forest Weather Index features are taken
train_features_three = X_train[['FFMC', 'DMC', 'DC', 'ISI']].copy()
test_features_three = X_test[['FFMC', 'DMC', 'DC', 'ISI']].copy()

# Naive Bayes Regressor

# Training using Naive Bayes Regressor
clf_nb.fit(train_features_three, y_train)
# Testing using Naive Bayes Regressor
predictions_nb_three = clf_nb.predict(test_features_three)

# RootMeanSquared Error Calculation for Naive Bayes Regressor
print("\n\nFor FWI features using naive_bayes_ridge")
meanSquaredError_nb_three = mean_squared_error(y_test, predictions_nb_three)
#print("MSE:", meanSquaredError_nb_three)
rootMeanSquaredError_nb_three = sqrt(meanSquaredError_nb_three)
print("RMSE:", rootMeanSquaredError_nb_three)

# MeanAbsolute Error Calculation for Naive Bayes Regressor
absolute_error_nb_three = mean_absolute_error(y_test, predictions_nb_three)
print("Absolute error is:", absolute_error_nb_three)



# Linear Regression

# Training using Linear Regression
clf_mr.fit(train_features_three, y_train)
# Testing using Linear Regression
predictions_mr_three = clf_mr.predict(test_features_three)

# RootMeanSquared Error Calculation for Linear Regression
print("\n\nFor FWI features using Linear Regression")
meanSquaredError_mr_three = mean_squared_error(y_test, predictions_mr_three)
#print("MSE:", meanSquaredError_mr_three)
rootMeanSquaredError_mr_three = sqrt(meanSquaredError_mr_three)
print("RMSE:", rootMeanSquaredError_mr_three)

# MeanAbsolute Error Calculation for Linear Regression
absolute_error_mr_three = mean_absolute_error(y_test, predictions_mr_three)
print("Absolute error is:", absolute_error_mr_three)



# Decision Trees Regressor

# Training using Decision Trees Regressor
clf_dt.fit(train_features_three, y_train)
# Testing using Decision Trees Regressor
predictions_dt_three = clf_dt.predict(test_features_three)

# RootMeanSquared Error Calculation for Decision Trees Regressor
print("\n\nFor FWI features using Decision Trees Regressor")
meanSquaredError_dt_three = mean_squared_error(y_test, predictions_dt_three)
#print("MSE:", meanSquaredError_dt_three)
rootMeanSquaredError_dt_three = sqrt(meanSquaredError_dt_three)
print("RMSE:", rootMeanSquaredError_dt_three)

# MeanAbsolute Error Calculation for Decision Trees Regressor
absolute_error_dt_three = mean_absolute_error(y_test, predictions_dt_three)
print("Absolute error is:", absolute_error_dt_three)



# Random Forest Regressor

# Training using Random Forest Regressor
clf_rf.fit(train_features_three, y_train)
# Testing using Random Forest Regressor
predictions_rf_three = clf_rf.predict(test_features_three)

# RootMeanSquared Error Calculation for Random Forest Regressor
print("\n\nFor FWI features using Random Forest Regressor")
meanSquaredError_rf_three = mean_squared_error(y_test, predictions_rf_three)
#print("MSE:", meanSquaredError_rf_three)
rootMeanSquaredError_rf_three = sqrt(meanSquaredError_rf_three)
print("RMSE:", rootMeanSquaredError_rf_three)

# MeanAbsolute Error Calculation for Random Forest Regressor
absolute_error_rf_three = mean_absolute_error(y_test, predictions_rf_three)
print("Absolute error is:", absolute_error_rf_three)



# MLP Regressor

# Training using MLP Regressor
clf_nn.fit(train_features_three, y_train)
# Testing using MLP Regressor
predictions_nn_three = clf_nn.predict(test_features_three)

# RootMeanSquared Error Calculation for MLP Regressor
print("\n\nFor FWI features using MLP Regressor")
meanSquaredError_nn_three = mean_squared_error(y_test, predictions_nn_three)
#print("MSE:", meanSquaredError_nn_three)
rootMeanSquaredError_nn_three = sqrt(meanSquaredError_nn_three)
print("RMSE:", rootMeanSquaredError_nn_three)

# MeanAbsolute Error Calculation for MLP Regressor
absolute_error_nn_three = mean_absolute_error(y_test, predictions_nn_three)
print("Absolute error is:", absolute_error_nn_three)



# SVM Regressor

# Training using SVM Regressor
clf_svm.fit(train_features_three, y_train)
# Testing using SVM Regressor
predictions_svm_three = clf_svm.predict(test_features_three)

# RootMeanSquared Error Calculation for SVM Regressor
print("\n\nFor FWI features using SVM Regressor")
meanSquaredError_svm_three = mean_squared_error(y_test, predictions_svm_three)
#print("MSE:", meanSquaredError_svm_three)
rootMeanSquaredError_svm_three = sqrt(meanSquaredError_svm_three)
print("RMSE:", rootMeanSquaredError_svm_three)

# MeanAbsolute Error Calculation for SVM Regressor
absolute_error_svm_three = mean_absolute_error(y_test, predictions_svm_three)
print("Absolute error is:", absolute_error_svm_three)


# In[6]:


# for M (using only four weather conditions)
train_features_four = X_train[[ 'temp', 'RH', 'wind', 'rain']].copy()
test_features_four = X_test[['temp', 'RH', 'wind', 'rain']].copy()

# Naive Bayes Regressor

# Training using Naive Bayes Regressor
clf_nb.fit(train_features_four, y_train)
# Testing using Naive Bayes Regressor
predictions_nb_four = clf_nb.predict(test_features_four)

# RootMeanSquared Error Calculation for Naive Bayes Regressor
print("\n\nFor M features using naive_bayes_ridge")
meanSquaredError_nb_four = mean_squared_error(y_test, predictions_nb_four)
#print("MSE:", meanSquaredError_nb_four)
rootMeanSquaredError_nb_four = sqrt(meanSquaredError_nb_four)
print("RMSE:", rootMeanSquaredError_nb_four)

# MeanAbsolute Error Calculation for Naive Bayes Regressor
absolute_error_nb_four = mean_absolute_error(y_test, predictions_nb_four)
print("Absolute error is:", absolute_error_nb_four)



# Linear Regression

# Training using Linear Regression
clf_mr.fit(train_features_four, y_train)
# Testing using Linear Regression
predictions_mr_four = clf_mr.predict(test_features_four)

# RootMeanSquared Error Calculation for Linear Regression
print("\n\nFor M features using Linear Regression")
meanSquaredError_mr_four = mean_squared_error(y_test, predictions_mr_four)
#print("MSE:", meanSquaredError_mr_four)
rootMeanSquaredError_mr_four = sqrt(meanSquaredError_mr_four)
print("RMSE:", rootMeanSquaredError_mr_four)

# MeanAbsolute Error Calculation for Linear Regression
absolute_error_mr_four = mean_absolute_error(y_test, predictions_mr_four)
print("Absolute error is:", absolute_error_mr_four)



# Decision Trees Regressor

# Training using Decision Trees Regressor
clf_dt.fit(train_features_four, y_train)
# Testing using Decision Trees Regressor
predictions_dt_four = clf_dt.predict(test_features_four)

# RootMeanSquared Error Calculation for Decision Trees Regressor
print("\n\nFor M features using Decision Trees Regressor")
meanSquaredError_dt_four = mean_squared_error(y_test, predictions_dt_four)
#print("MSE:", meanSquaredError_dt_four)
rootMeanSquaredError_dt_four = sqrt(meanSquaredError_dt_four)
print("RMSE:", rootMeanSquaredError_dt_four)

# MeanAbsolute Error Calculation for Decision Trees Regressor
absolute_error_dt_four = mean_absolute_error(y_test, predictions_dt_four)
print("Absolute error is:", absolute_error_dt_four)



# Random Forest Regressor

# Training using Random Forest Regressor
clf_rf.fit(train_features_four, y_train)
# Testing using Random Forest Regressor
predictions_rf_four = clf_rf.predict(test_features_four)

# RootMeanSquared Error Calculation for Random Forest Regressor
print("\n\nFor M features using Random Forest Regressor")
meanSquaredError_rf_four = mean_squared_error(y_test, predictions_rf_four)
#print("MSE:", meanSquaredError_rf_four)
rootMeanSquaredError_rf_four = sqrt(meanSquaredError_rf_four)
print("RMSE:", rootMeanSquaredError_rf_four)

# MeanAbsolute Error Calculation for Random Forest Regressor
absolute_error_rf_four = mean_absolute_error(y_test, predictions_rf_four)
print("Absolute error is:", absolute_error_rf_four)



# MLP Regressor

# Training using MLP Regressor
clf_nn.fit(train_features_four, y_train)
# Testing using MLP Regressor
predictions_nn_four = clf_nn.predict(test_features_four)

# RootMeanSquared Error Calculation for MLP Regressor
print("\n\nFor M features using MLP Regressor")
meanSquaredError_nn_four = mean_squared_error(y_test, predictions_nn_four)
#print("MSE:", meanSquaredError_nn_four)
rootMeanSquaredError_nn_four = sqrt(meanSquaredError_nn_four)
print("RMSE:", rootMeanSquaredError_nn_four)

# MeanAbsolute Error Calculation for MLP Regressor
absolute_error_nn_four = mean_absolute_error(y_test, predictions_nn_four)
print("Absolute error is:", absolute_error_nn_four)



# SVM Regressor

# Training using SVM Regressor
clf_svm.fit(train_features_four, y_train)
# Testing using SVM Regressor
predictions_svm_four = clf_svm.predict(test_features_four)

# RootMeanSquared Error Calculation for SVM Regressor
print("\n\nFor M features using SVM Regressor")
meanSquaredError_svm_four = mean_squared_error(y_test, predictions_svm_four)
#print("MSE:", meanSquaredError_svm_four)
rootMeanSquaredError_svm_four = sqrt(meanSquaredError_svm_four)
print("RMSE:", rootMeanSquaredError_svm_four)

# MeanAbsolute Error Calculation for SVM Regressor
absolute_error_svm_four = mean_absolute_error(y_test, predictions_svm_four)
print("Absolute error is:", absolute_error_svm_four)

