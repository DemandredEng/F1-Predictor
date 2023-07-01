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

y_train = train.finishing_position
y_test21 = test21.finishing_position
y_test22 = test22.finishing_position

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

def Linear_Regression_Predictor(x_train, testyr, x_testyr, titl):
    lin_reg_model = LinearRegression(fit_intercept= True)
    lin_reg_model.fit(x_train, y_train)
    lin_reg_model_pred = testyr[['season', 'round', 'driver','finishing_position']].copy()
    lin_reg_model_pred['Predicted'] = lin_reg_model.predict(x_testyr)
    lin_reg_model_pred['Predicted Position'] = lin_reg_model_pred.groupby(['season', 'round'])['Predicted'].rank()
    performance_metric(lin_reg_model_pred, titl)

Linear_Regression_Predictor(xv3b_train, test21, xv3b_test21, 'Linear Regression V3b, 2021')
Linear_Regression_Predictor(xv3b_train, test22, xv3b_test22, 'Linear Regression V3b, 2022')

# MLP MODEL

def objective_mlp(trial):
    params = {
        'layer_1': trial.suggest_int('layer_1', 65, 95, step=5),
        'layer_2': trial.suggest_int('layer_2', 15, 55, step=5),
        'layer_3': trial.suggest_int('layer_3', 20, 60, step=5),
        'layer_4': trial.suggest_int('layer_4', 5, 25, step=5),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.9996, step=0.0005),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'alpha': trial.suggest_float('alpha', 1e-6, 1e-3),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])}
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(params['layer_1'], params['layer_2'], params['layer_3'],params['layer_4']),
        learning_rate_init=params['learning_rate_init'], learning_rate= params['learning_rate'],
        alpha = params['alpha'], activation=params['activation'], random_state=42, max_iter=1000)
    
    mlp_model.fit(xv3b_train, y_train)
    scores = -cross_val_score(mlp_model, xv3b_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective_mlp, n_trials=300)

best_mlp_params = study.best_params
print("Best MLP Parameters:")
print(best_mlp_params)

mlp_model = MLPRegressor(
        hidden_layer_sizes=(best_mlp_params['layer_1'], best_mlp_params['layer_2'], best_mlp_params['layer_3'],best_mlp_params['layer_4']),
        learning_rate_init=best_mlp_params['learning_rate_init'], learning_rate= best_mlp_params['learning_rate'],
        alpha = best_mlp_params['alpha'], activation=best_mlp_params['activation'], random_state=42, max_iter=1000)

mlp_model.fit(xv3b_train, y_train)

def MLP_Predictor(testyr, x_testyr, titl):
    mlp_pred = testyr[['season', 'round', 'driver','finishing_position']].copy()
    mlp_pred['Predicted'] = mlp_model.predict(x_testyr)
    mlp_pred['Predicted Position'] = mlp_pred.groupby(['season', 'round'])['Predicted'].rank()
    performance_metric(mlp_pred, titl)

MLP_Predictor(test21, xv3b_test21, 'MLP V3b, 2021')
MLP_Predictor(test22, xv3b_test22, 'MLP V3b, 2022')

print(model_performance)
model_performance.to_excel('Model_Performance_V3b_Final.xlsx', index=False)