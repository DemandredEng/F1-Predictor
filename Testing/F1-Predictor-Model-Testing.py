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

def MLP_model_creator(x_train, x_test20, x_test21, x_test22, titl20, titl21, titl22, num_tri):

    def objective_mlp(trial):
        params = {
            'layer_1': trial.suggest_int('layer_1', 55, 95, step=5),
            'layer_2': trial.suggest_int('layer_2', 10, 55, step=5),
            'layer_3': trial.suggest_int('layer_3', 20, 60, step=5),
            'layer_4': trial.suggest_int('layer_4', 5, 35, step=5),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.00001, 0.99951, step=0.0005),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1),
            'activation': trial.suggest_categorical('activation', ['tanh', 'relu'])}
        
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(params['layer_1'], params['layer_2'], params['layer_3'],params['layer_4']),
            learning_rate_init=params['learning_rate_init'], learning_rate= params['learning_rate'],
            alpha = params['alpha'], activation=params['activation'], random_state=42, max_iter=1000)
        
        mlp_model.fit(xv3b_train, y_train)
        scores = -cross_val_score(mlp_model, xv3b_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_mlp, n_trials=num_tri)

    best_mlp_params = study.best_params
    best_iteration_score = study.best_value

    print("Best MLP Parameters:")
    print(best_mlp_params)
    print("Best Iteration Score:", best_iteration_score)

    data = {"Parameters": [best_mlp_params], "Best Iteration Score": [best_iteration_score]}
    df = pd.DataFrame(data)

    file_name = f"best_params_{titl21}.xlsx"
    df.to_excel(file_name, index=False)      #titl20, titl21, titl22 same model

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(best_mlp_params['layer_1'], best_mlp_params['layer_2'], best_mlp_params['layer_3'],best_mlp_params['layer_4']),
        learning_rate_init=best_mlp_params['learning_rate_init'], learning_rate= best_mlp_params['learning_rate'],
        alpha = best_mlp_params['alpha'], activation=best_mlp_params['activation'], random_state=42, max_iter=1000)
    
    mlp_model.fit(x_train, y_train)
    MLP_Predictor(mlp_model, test20, x_test20, titl20)
    MLP_Predictor(mlp_model, test21, x_test21, titl21)
    MLP_Predictor(mlp_model, test22, x_test22, titl22)

MLP_model_creator(xv3b_train, xv3b_test20, xv3b_test21, xv3b_test22, 'MLP V3b, 2020', 'MLP V3b, 2021','MLP V3b, 2022', 200)

def XGB_Predictor(xgb_model, testyr, x_testyr, titl):
    xgb_pred = testyr[['season', 'round', 'driver','finishing_position']].copy()
    xgb_pred['Predicted'] = xgb_model.predict(x_testyr)
    xgb_pred['Predicted Position'] = xgb_pred.groupby(['season', 'round'])['Predicted'].rank()
    performance_metric(xgb_pred, titl)
    '''
    file_name = f'{titl}_XG_model.xlsx'
    xgb_pred.to_excel(file_name, index=False)
    '''
def XGB_model_creator(x_train, x_test20,x_test21, x_test22, titl20, titl21, titl22, num_tri):

    def objective_xgb(trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

        xgb_model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42
        )

        xgb_model.fit(x_train, y_train)
        scores = -cross_val_score(xgb_model, x_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgb, n_trials=num_tri)

    best_xgb_params = study.best_params
    best_iteration_score = study.best_value

    print("Best XGBoost Parameters:")
    print(best_xgb_params)
    print("Best Iteration Score:", best_iteration_score)

    data = {"Parameters": [best_xgb_params], "Best Iteration Score": [best_iteration_score]}
    df = pd.DataFrame(data)

    file_name = f"best_params_{titl21}.xlsx"
    df.to_excel(file_name, index=False)      #titl20, titl21, titl22 same model

    xgb_model = xgb.XGBRegressor(
        n_estimators=best_xgb_params['n_estimators'],
        max_depth=best_xgb_params['max_depth'],
        learning_rate=best_xgb_params['learning_rate'],
        gamma=best_xgb_params['gamma'],
        subsample=best_xgb_params['subsample'],
        colsample_bytree=best_xgb_params['colsample_bytree'],
        random_state=42
    )

    xgb_model.fit(x_train, y_train)
    XGB_Predictor(xgb_model, test20, x_test20, titl20)
    XGB_Predictor(xgb_model, test21, x_test21, titl21)
    XGB_Predictor(xgb_model, test22, x_test22, titl22)

XGB_model_creator(xv3b_train, xv3b_test20, xv3b_test21, xv3b_test22, 'XGBoost V3b, 2020', 'XGBoost V3b, 2021', 'XGBoost V3b, 2022', 200)

print(model_performance)
model_performance.to_excel('Model_Performance.xlsx', index=False)