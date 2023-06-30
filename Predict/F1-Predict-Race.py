import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Data collected using the Ergast F1 API and the official F1 website

data = pd.read_csv('df_f1.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

df.drop(['driver_wins_prev_season', 'constructor_wins_prev_season', 'driver_age'], axis=1, inplace = True)

# Version 3b - For my reference

dfV3b = df.copy()
grouped = dfV3b.groupby(['season', 'round'])
highest_values = grouped[['driver_points', 'constructor_points']].transform('max')
dfV3b['driver_points_percentage'] = dfV3b['driver_points'] / highest_values['driver_points'] 
dfV3b['constructor_points_percentage'] = dfV3b['constructor_points'] / highest_values['constructor_points'] 
dfV3b['driver_points_percentage'] = dfV3b['driver_points_percentage'].fillna(0)
dfV3b['constructor_points_percentage'] = dfV3b['constructor_points_percentage'].fillna(0)

# prev season
grouped = dfV3b.groupby(['season', 'round'])
highest_values = grouped[['driver_points_prev_season', 'constructor_points_prev_season']].transform('max')
dfV3b['driver_points_prev_season_percentage'] = dfV3b['driver_points_prev_season'] / highest_values['driver_points_prev_season'] 
dfV3b['constructor_points_prev_season_percentage'] = dfV3b['constructor_points_prev_season'] / highest_values['constructor_points_prev_season'] 
dfV3b['driver_points_prev_season_percentage'] = dfV3b['driver_points_prev_season_percentage'].fillna(0)
dfV3b['constructor_points_prev_season_percentage'] = dfV3b['constructor_points_prev_season_percentage'].fillna(0)

columns_to_drop_V3b = ['driver_points', 'constructor_points', 'driver_points_prev_season', 'constructor_points_prev_season']
dfV3b.drop(columns_to_drop_V3b, axis=1, inplace=True)

train = dfV3b[dfV3b.season < 2020]
test20 = dfV3b[dfV3b['season'] == 2020]
test21 = dfV3b[dfV3b['season'] == 2021]
test22 = dfV3b[dfV3b['season'] == 2022]

xv3b_train = train.drop(['driver', 'nationality', 'constructor',  'finishing_position'], axis = 1)
xv3b_test20 = test20.drop(['driver', 'nationality', 'constructor',  'finishing_position'], axis=1)
xv3b_test21 = test21.drop(['driver', 'nationality', 'constructor',  'finishing_position'], axis=1)
xv3b_test22 = test22.drop(['driver', 'nationality', 'constructor',  'finishing_position'], axis=1)

y_train = train.finishing_position
y_test20 = test20.finishing_position
y_test21 = test21.finishing_position
y_test22 = test22.finishing_position

#xv3b_train.to_excel('xv3b_train.xlsx', index= False)

# RACE PREDICT INPUT

pred_race = pd.read_csv('predict-race.csv')
pred_race.drop(['driver_wins_prev_season', 'constructor_wins_prev_season', 'driver_age'], axis=1, inplace = True)

grouped = pred_race.groupby(['season', 'round'])
highest_values = grouped[['driver_points', 'constructor_points']].transform('max')
pred_race['driver_points_percentage'] = pred_race['driver_points'] / highest_values['driver_points'] 
pred_race['constructor_points_percentage'] = pred_race['constructor_points'] / highest_values['constructor_points'] 
pred_race['driver_points_percentage'] = pred_race['driver_points_percentage'].fillna(0)
pred_race['constructor_points_percentage'] = pred_race['constructor_points_percentage'].fillna(0)

# prev season
grouped = pred_race.groupby(['season', 'round'])
highest_values = grouped[['driver_points_prev_season', 'constructor_points_prev_season']].transform('max')
pred_race['driver_points_prev_season_percentage'] = pred_race['driver_points_prev_season'] / highest_values['driver_points_prev_season'] 
pred_race['constructor_points_prev_season_percentage'] = pred_race['constructor_points_prev_season'] / highest_values['constructor_points_prev_season'] 
pred_race['driver_points_prev_season_percentage'] = pred_race['driver_points_prev_season_percentage'].fillna(0)
pred_race['constructor_points_prev_season_percentage'] = pred_race['constructor_points_prev_season_percentage'].fillna(0)

columns_to_drop_V3b = ['driver_points', 'constructor_points', 'driver_points_prev_season', 'constructor_points_prev_season']
pred_race.drop(columns_to_drop_V3b, axis=1, inplace=True)
x_pred_race = pred_race.drop(['driver', 'nationality', 'constructor'], axis=1)

# LINEAR REGRESSION MODEL, FINAL PREDICTED POSITIONS RANKED.

def Linear_Regression_Predictor(x_train, testyr, x_testyr, titl):

    lin_reg_model = LinearRegression(fit_intercept= True)
    lin_reg_model.fit(x_train, y_train)
    lin_reg_model_pred = testyr[['season', 'round', 'driver']].copy()
    lin_reg_model_pred['Predicted'] = lin_reg_model.predict(x_testyr)
    lin_reg_model_pred['Predicted Position'] = lin_reg_model_pred.groupby(['season', 'round'])['Predicted'].rank()
    lin_reg_model_pred.to_excel('Linear Regression Predictions.xlsx', index=False)

Linear_Regression_Predictor(xv3b_train, pred_race, x_pred_race, 'Linear Regression Predictions')


# MLP REGRESSOR

def MLP_Predictor(mlp_model, testyr, x_testyr, titl):
    mlp_pred = testyr[['season', 'round', 'driver']].copy()
    mlp_pred['Predicted'] = mlp_model.predict(x_testyr)
    mlp_pred['Predicted Position'] = mlp_pred.groupby(['season', 'round'])['Predicted'].rank()
    mlp_pred.to_excel('MLP Regressor Predictions.xlsx', index=False)

def MLP_model_creator(hidden_layer, learning_rt_init, learning_rt, alph, actvtn, x_train, testyr, x_testyr, titl):
    mlp_model = MLPRegressor(hidden_layer_sizes= hidden_layer, learning_rate_init= learning_rt_init, learning_rate = learning_rt,
                             alpha= alph, activation= actvtn, random_state= 42, max_iter = 1000)
    mlp_model.fit(x_train, y_train)
    MLP_Predictor(mlp_model, testyr, x_testyr, titl)

MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, pred_race, x_pred_race, 'MLP Regressor Predictions')

