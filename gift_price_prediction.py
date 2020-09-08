# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:12:53 2020

@author: India
"""

#Importing the required libraries

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import date,datetime,timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#reading the training dataset
dataset=pd.read_csv("F:\\Machine Learning\\HackerEarth\\dataset\\train.csv") 

#loading the test dataset
test_dataset=pd.read_csv("F:\\Machine Learning\\HackerEarth\\dataset\\test.csv") 

#Performing exploratory data analytics to understand feature corelations and interactions
prof = ProfileReport(dataset)
prof.to_file(output_file='F:\\Machine Learning\\HackerEarth\\dataset\\gift_prices_EDA.html')

#Converting the features containing timestamp from string to datetime
dataset['uk_date1'] = pd.to_datetime(dataset['uk_date1'], infer_datetime_format=True)
dataset['uk_date2'] = pd.to_datetime(dataset['uk_date2'], infer_datetime_format=True)
dataset['instock_date'] = pd.to_datetime(dataset['instock_date'], infer_datetime_format=True)
dataset['stock_update_date'] = pd.to_datetime(dataset['stock_update_date'], infer_datetime_format=True)


#Adding new features in order to calculate the time difference between the datetime fields in seconds in order to effectively use them for model building

dataset['buy_wait_1']=abs(dataset['uk_date1']-dataset['instock_date']).dt.total_seconds()
dataset['buy_wait_2']=abs(dataset['uk_date2']-dataset['instock_date']).dt.total_seconds()
dataset['stock_update_tm']=abs(dataset['stock_update_date']-dataset['instock_date']).dt.total_seconds()


#performing the same string to datetime conversions for the test dataset

test_dataset['uk_date1'] = pd.to_datetime(test_dataset['uk_date1'], infer_datetime_format=True)
test_dataset['uk_date2'] = pd.to_datetime(test_dataset['uk_date2'], infer_datetime_format=True)
test_dataset['instock_date'] = pd.to_datetime(test_dataset['instock_date'], infer_datetime_format=True)
test_dataset['stock_update_date'] = pd.to_datetime(test_dataset['stock_update_date'], infer_datetime_format=True)

test_dataset['buy_wait_1']=abs(test_dataset['uk_date1']-test_dataset['instock_date']).dt.total_seconds()
test_dataset['buy_wait_2']=abs(test_dataset['uk_date2']-test_dataset['instock_date']).dt.total_seconds()
test_dataset['stock_update_tm']=abs(test_dataset['stock_update_date']-test_dataset['instock_date']).dt.total_seconds()



#Choosing required  features for model building based on EDA

features=['lsg_5','stock_update_tm','buy_wait_1', 'buy_wait_2', 'lsg_2', 'gift_category', 'lsg_4', 'gift_type', 'lsg_1', 'lsg_3', 'gift_cluster','lsg_6', 'is_discounted']

train=dataset[features]
y=dataset['price']

test=test_dataset[features]


#Scaling the training data
scaler = StandardScaler()

scaler.fit(train)
train = scaler.transform(train)

#scaling the test dataset

scaler.fit(test)
test=scaler.transform(test)

#Choosing the random forest regressor to build the model
model=RandomForestRegressor(n_jobs=-1)

train_X, val_X, train_y, val_y = train_test_split(train, y,test_size=0.1, random_state = 0)
#Trying different n_estimators values for hyper parameter tunig to yeild the best scores
n_estimators = np.arange(10, 400, 10)
scores = []
for n in n_estimators:
    model.set_params(n_estimators=n)
    model.fit(train_X, train_y)
    scores.append(model.score(val_X, val_y))
    
#Ploting the scores for each estimator value    
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)

#building an updated model with the favoured hyper parameter values

pricing_model=RandomForestRegressor(n_jobs=-1,n_estimators=350)

#fitting the model
pricing_model.fit(train, y)

#predicting the prices based on the trained model
predictions = pricing_model.predict(test)
model_score=pricing_model.score(val_X, val_y)


model_MSE=mean_squared_error(val_y,predictions)**0.5
model_MAE=mean_absolute_error(val_y,predictions)


m2_mae=mean_absolute_error(val_y,predictions)

test_gift_id=(test_dataset['gift_id'])

type(predictions)
submission=pd.DataFrame(test_gift_id)
submission['price']=predictions
submission_file=submission.to_csv("F:\\Machine Learning\\HackerEarth\\dataset\\submission.csv",index = None, header=True)


