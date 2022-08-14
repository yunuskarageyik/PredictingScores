# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 00:46:07 2022

@author: User
"""
# İmporting Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datas = pd.read_csv("score.csv")
# file download link = https://www.kaggle.com/datasets/himanshunakrani/student-study-hours


hours = datas.iloc[:,0:1].values # Getting the time data.
scores = datas.iloc[:,1:].values # Getting the score.

# Predicting scores with linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(hours,scores) 

# Visualization
plt.scatter(hours,scores,color="red")
plt.plot(hours,lin_reg.predict(hours),color="blue")
