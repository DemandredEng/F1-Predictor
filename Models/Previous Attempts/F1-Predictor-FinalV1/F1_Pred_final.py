import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('final_df.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

scaler = MinMaxScaler()
driver_age = df['driver_age'].values.reshape(-1, 1)
scaled_driver_age = scaler.fit_transform(driver_age)
df['driver_age'] = scaled_driver_age

#V2 - Points and Standings
train = df[df.season < 2021]
test21 = df[df['season'] == 2021]
test22 = df[df['season'] == 2022]
y_train = train.finishing_position
y_test21 = test21.finishing_position
y_test22 = test22.finishing_position

xv2_train = train.drop(['driver' , 'finishing_position'], axis = 1)
xv2_test21 = test21.drop(['driver', 'finishing_position'], axis=1)
xv2_test22 = test22.drop(['driver', 'finishing_position'], axis=1)

#V3c - No points, only standings
columns_to_drop_V3c = ['driver_points', 'constructor_points']
dfV3c = df.copy()
dfV3c.drop(columns_to_drop_V3c, axis=1, inplace=True)

train = dfV3c[dfV3c.season < 2021]
test21 = dfV3c[dfV3c['season'] == 2021]
test22 = dfV3c[dfV3c['season'] == 2022]

xv3c_train = train.drop(['driver' , 'finishing_position'], axis = 1)
xv3c_test21 = test21.drop(['driver', 'finishing_position'], axis=1)
xv3c_test22 = test22.drop(['driver', 'finishing_position'], axis=1)

#V3ab -- V3b - Points % and Standings
dfV3b = df.copy()
grouped = dfV3b.groupby(['season', 'round'])
highest_values = grouped[['driver_points', 'constructor_points']].transform('max')
dfV3b['driver_points_percentage'] = dfV3b['driver_points'] / highest_values['driver_points'] 
dfV3b['constructor_points_percentage'] = dfV3b['constructor_points'] / highest_values['constructor_points'] 
dfV3b['driver_points_percentage'] = dfV3b['driver_points_percentage'].fillna(0)
dfV3b['constructor_points_percentage'] = dfV3b['constructor_points_percentage'].fillna(0)
columns_to_drop_V3b = ['driver_points', 'constructor_points']
dfV3b.drop(columns_to_drop_V3b, axis=1, inplace=True)

train = dfV3b[dfV3b.season < 2021]
test21 = dfV3b[dfV3b['season'] == 2021]
test22 = dfV3b[dfV3b['season'] == 2022]

xv3b_train = train.drop(['driver' , 'finishing_position'], axis = 1)
xv3b_test21 = test21.drop(['driver', 'finishing_position'], axis=1)
xv3b_test22 = test22.drop(['driver', 'finishing_position'], axis=1)

#V3a - Only Points %
dfV3a = dfV3b.copy()
columns_to_drop_V3a = ['driver_standings_pos', 'constructor_standings_pos']
dfV3a.drop(columns_to_drop_V3a, axis=1, inplace=True)

train = dfV3a[dfV3a.season < 2021]
test21 = dfV3a[dfV3a['season'] == 2021]
test22 = dfV3a[dfV3a['season'] == 2022]

xv3a_train = train.drop(['driver' , 'finishing_position'], axis = 1)
xv3a_test21 = test21.drop(['driver', 'finishing_position'], axis=1)
xv3a_test22 = test22.drop(['driver', 'finishing_position'], axis=1)

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
    performance_metric(lin_reg_model_pred, titl)
    

Linear_Regression_Predictor(xv2_train, test21, xv2_test21, 'Linear Regression V2, 2021')
Linear_Regression_Predictor(xv2_train, test22, xv2_test22, 'Linear Regression V2, 2022')
Linear_Regression_Predictor(xv3a_train, test21, xv3a_test21, 'Linear Regression V3a, 2021')
Linear_Regression_Predictor(xv3a_train, test22, xv3a_test22, 'Linear Regression V3a, 2022')
Linear_Regression_Predictor(xv3b_train, test21, xv3b_test21, 'Linear Regression V3b, 2021')
Linear_Regression_Predictor(xv3b_train, test22, xv3b_test22, 'Linear Regression V3b, 2022')
Linear_Regression_Predictor(xv3c_train, test21, xv3c_test21, 'Linear Regression V3c, 2021')
Linear_Regression_Predictor(xv3c_train, test22, xv3c_test22, 'Linear Regression V3c, 2022')

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
    
MLP_model_creator([90,45,30,15], 0.0026, 'adaptive', 0.00019784360231304976, 'relu',xv2_train, test21, xv2_test21, 'MLP V2a, 2021')
MLP_model_creator([90,45,30,15], 0.0026, 'adaptive', 0.00019784360231304976, 'relu',xv2_train, test22, xv2_test22, 'MLP V2a, 2022')
MLP_model_creator([85,25,40,10], 0.0001, 'constant', 0.000781851065130057, 'relu',xv2_train, test21, xv2_test21, 'MLP V2b, 2021')
MLP_model_creator([85,25,40,10], 0.0001, 'constant', 0.000781851065130057, 'relu',xv2_train, test22, xv2_test22, 'MLP V2b, 2022')
MLP_model_creator([80,20,50,10], 1e-05, 'constant', 0.0002755041294744065, 'relu',xv2_train, test21, xv2_test21, 'MLP V2c, 2021')
MLP_model_creator([80,20,50,10], 1e-05, 'constant', 0.0002755041294744065, 'relu',xv2_train, test22, xv2_test22, 'MLP V2c, 2022')

MLP_model_creator([65,15,35,15], 0.0001, 'adaptive', 0.0005842727410236075, 'relu',xv3a_train, test21, xv3a_test21, 'MLP V3a, 2021')
MLP_model_creator([65,15,35,15], 0.0001, 'adaptive', 0.0005842727410236075, 'relu',xv3a_train, test22, xv3a_test22, 'MLP V3a, 2022')
MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, test21, xv3b_test21, 'MLP V3b, 2021')
MLP_model_creator([65,45,35,25], 0.0011, 'adaptive', 0.0005376090978671149, 'relu',xv3b_train, test22, xv3b_test22, 'MLP V3b, 2022')
MLP_model_creator([90,45,40,15], 0.0006000000000000001, 'constant', 0.0007383389329083249, 'relu',xv3c_train, test21, xv3c_test21, 'MLP V3c, 2021')
MLP_model_creator([90,45,40,15], 0.0006000000000000001, 'constant', 0.0007383389329083249, 'relu',xv3c_train, test22, xv3c_test22, 'MLP V3c, 2022')

print(model_performance)
model_performance.to_excel('Model_Performance_Final.xlsx', index=False)