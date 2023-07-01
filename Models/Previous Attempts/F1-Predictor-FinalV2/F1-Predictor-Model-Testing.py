import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor

# Data collected using the Ergast F1 API and the official F1 website

data = pd.read_csv('df_f1.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

df.drop(['driver_wins_prev_season', 'constructor_wins_prev_season'], axis=1, inplace = True)

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

# Scaling driver's age
scaler = MinMaxScaler()
train_age = xv3b_train['driver_age'].values.reshape(-1,1)
xv3b_train['driver_age'] = scaler.fit_transform(train_age)

test20_age = xv3b_test20['driver_age'].values.reshape(-1,1)
xv3b_test20['driver_age'] = scaler.transform(test20_age)
test21_age = xv3b_test21['driver_age'].values.reshape(-1,1)
xv3b_test21['driver_age'] = scaler.transform(test21_age)
test22_age = xv3b_test22['driver_age'].values.reshape(-1,1)
xv3b_test22['driver_age'] = scaler.transform(test22_age)

tscv = TimeSeriesSplit(n_splits = 20)
model_performance = pd.DataFrame(columns=['Model', 'Mean Absolute Error', 'Root Mean Squared Error', 'Percentage Correct positions',
                                          'Percentage Correct Wins', 'Percentage Correct 2nd Place', 'Percentage Correct 3rd Place'])

def performance_metric(df, model_name):
    global model_performance
    y_true = df['finishing_position']
    y_pred = df['Predicted Position']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correct_positions = (y_true == y_pred).mean() * 100
    correct_pred_wins = ((y_true == 1) & (y_pred == 1)).sum()
    total_first_positions = (y_true == 1).sum()
    corr_wins = (correct_pred_wins / total_first_positions) * 100

    correct_pred_2nd = ((y_true == 2) & (y_pred == 2)).sum()
    total_second_positions = (y_true == 2).sum()
    corr_2nd = (correct_pred_2nd / total_second_positions) * 100
    correct_pred_3rd = ((y_true == 3) & (y_pred == 3)).sum()
    total_third_positions = (y_true == 3).sum()
    corr_3rd = (correct_pred_3rd / total_third_positions) * 100

    model_performance.loc[len(model_performance)] = {'Model': model_name, 'Mean Absolute Error': mae,
                                                     'Root Mean Squared Error': rmse,
                                                     'Percentage Correct positions': correct_positions,
                                                     'Percentage Correct Wins': corr_wins,
                                                     'Percentage Correct 2nd Place': corr_2nd,
                                                     'Percentage Correct 3rd Place': corr_3rd}
    return

# LINEAR REGRESSION MODEL, FINAL PREDICTED POSITIONS RANKED.

def Linear_Regression_Predictor(x_train, testyr, x_testyr, titl):

    lin_reg_model = LinearRegression(fit_intercept= True)
    lin_reg_model.fit(x_train, y_train)
    lin_reg_model_pred = testyr[['season', 'round', 'driver','finishing_position']].copy()
    lin_reg_model_pred['Predicted'] = lin_reg_model.predict(x_testyr)
    lin_reg_model_pred['Predicted Position'] = lin_reg_model_pred.groupby(['season', 'round'])['Predicted'].rank()

    '''
     # If you want to see predicted positions
    file_name = f'{titl}_linreg_model.xlsx'
    lin_reg_model_pred.to_excel(file_name, index=False)
    '''
    
    performance_metric(lin_reg_model_pred, titl)


Linear_Regression_Predictor(xv3b_train, test20, xv3b_test20, 'Linear Regression V3b, 2020')
Linear_Regression_Predictor(xv3b_train, test21, xv3b_test21, 'Linear Regression V3b, 2021')
Linear_Regression_Predictor(xv3b_train, test22, xv3b_test22, 'Linear Regression V3b, 2022')

# MLP REGRESSOR

def MLP_Predictor(mlp_model, testyr, x_testyr, titl):
    mlp_pred = testyr[['season', 'round', 'driver','finishing_position']].copy()
    mlp_pred['Predicted'] = mlp_model.predict(x_testyr)
    mlp_pred['Predicted Position'] = mlp_pred.groupby(['season', 'round'])['Predicted'].rank()
    performance_metric(mlp_pred, titl)

def MLP_model_creator(hidden_layer, learning_rt_init, learning_rt, alph, actvtn, x_train, testyr, x_testyr, titl):
    mlp_model = MLPRegressor(hidden_layer_sizes= hidden_layer, learning_rate_init= learning_rt_init, learning_rate = learning_rt,
                             alpha= alph, activation= actvtn, random_state= 42, max_iter = 1000)
    mlp_model.fit(x_train, y_train)
    MLP_Predictor(mlp_model, testyr, x_testyr, titl)


MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, test20, xv3b_test20, 'MLP V3b, 2020')
MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, test21, xv3b_test21, 'MLP V3b, 2021')
MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, test22, xv3b_test22, 'MLP V3b, 2022')

print(model_performance)
model_performance.to_excel('Model_Performance.xlsx', index=False)

