# coding:utf-8

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer
from scipy import stats
import os

plt.rcParams.update({'figure.max_open_warning': 0})
# load_dataset
with open(r'C:\Users\zugle\Desktop\Python_code\steam\zhengqi_train.txt') as fr:
    data_train = pd.read_table(fr, sep="\t")
    print('data_train.shape=', data_train.shape)

with open(r'C:\Users\zugle\Desktop\Python_code\steam\zhengqi_test.txt') as fr_test:
    data_test = pd.read_table(fr_test, sep="\t")
    print('data_test.shape=', data_test.shape)
# merge train_set and test_set  add origin
data_train["oringin"] = "train"
data_test["oringin"] = "test"
data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True)
# View data
print('data_all.shape=', data_all.shape)
# Explore feature distibution
fig = plt.figure(figsize=(6, 6))
for column in data_all.columns[0:-2]:
    g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade=True)
    g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], color="Blue", shade=True)
    g.set(xlabel=column, ylabel='Frequency')
    g = g.legend(["train", "test"])
    plt.show()
