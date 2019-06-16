# Data-Mining-Approach-to-Predict-Forest-Fires

Referred Dataset and paper: https://archive.ics.uci.edu/ml/datasets/Forest+Fires

Problem statement: Given spatial, temporal, FWI Components and meteorological inputs of the recent real-world data collected from the northeast regions of Portugal, use Data Mining (DM) approaches to predict the burned area
of forest fires.

Data set: The dataset in the above referenced site is given in a csv format.
There are 12 features in total named:
X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area where
X, Y: spatial features
Month, day: temporal features
FFMC, DMC, DC, ISI: (FWI features)-Forest Weather Index
temp, RH, rain, wind: meteorological features


Difference in dataset which is given originally and the one we used forpredictions in our dataset:
 The original dataset contains two attributes named Month and day which are in string format so we convert them to integer by assigning numeric values to them.
(1-12) for Months and (1-7) for months.
We split our dataset into training and testing data(0.25) by using standard python library functions.

Preprocessing : We Standardize our input data as mentioned in the paper.
Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance).

Models Used:

1.Naive Bayes ridge using each of the below mentioned set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M: Meteorological features

2.Multiple regression(MR) using each of the below mentioned set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M: Meteorological features

3. Decision Tree Regressor(DT) for linear regression using each of the below
mentioned set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M: Meteorological features

4. .Random forest Regressor for linear regression using each of the below
mentioned set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M: Meteorological features

5. Multilayer Perceptron or linear regression using each of the below mentioned
set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M : meteorological features

6.Support Vector Machine for linear regression using each of the below
mentioned set of features.
a). STFWI: Spatial, Temporal, Forest Weather Index
b).STM: Spatial, Temporal, Meteorological
c).FWI: Forest Weather Index
d). M :meteorological features

Two types of errors used to compute the accuracy of the model.

1.MAD: Mean Absolute Deviation:
MAD = 1/N ×( Σi=1 i=n |yi – yi ^|)

2.RMSE: Root Mean Square Error.
RMSE = √( Σi=1i=n(yi− yi^)2/N )

In both metrics, lower values result in better predictive mod.
