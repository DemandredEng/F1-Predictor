import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('final_df.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

grouped = df.groupby(['season', 'round'])
highest_values = grouped[['driver_points', 'constructor_points']].transform('max')
df['driver_points_percentage'] = df['driver_points'] / highest_values['driver_points'] 
df['constructor_points_percentage'] = df['constructor_points'] / highest_values['constructor_points'] 
df['driver_points_percentage'] = df['driver_points_percentage'].fillna(0)
df['constructor_points_percentage'] = df['constructor_points_percentage'].fillna(0)
columns_to_drop = ['driver_points', 'constructor_points', 'driver_standings_pos', 'constructor_standings_pos']
df.drop(columns_to_drop, axis=1, inplace=True)

scaler = MinMaxScaler()
driver_age = df['driver_age'].values.reshape(-1, 1)
scaled_driver_age = scaler.fit_transform(driver_age)
df['driver_age'] = scaled_driver_age

train = df[df.season < 2021]
test = df[df['season'] > 2020]

x_train = train.drop(['driver' , 'finishing_position'], axis = 1)
y_train = train.finishing_position
x_test = test.drop(['driver', 'finishing_position'], axis=1)
y_test = test.finishing_position

tscv = TimeSeriesSplit(n_splits = 10)
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

lin_reg_model = LinearRegression(fit_intercept= True)
lin_reg_model.fit(x_train, y_train)
lin_reg_model_pred = test[['season', 'round', 'driver','finishing_position']].copy()
lin_reg_model_pred['Predicted'] = lin_reg_model.predict(x_test)
lin_reg_model_pred['Predicted Position'] = lin_reg_model_pred.groupby(['season', 'round'])['Predicted'].rank()

lin_reg_model_pred.to_excel('Lin_reg_model_pred.xlsx', index=False)

performance_metric(lin_reg_model_pred, 'Linear Regression')
print(model_performance)

# RANDOM FOREST MODEL
'''
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    rf_model = RandomForestRegressor(random_state=42, **params)
    scores = -cross_val_score(rf_model, x_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective_rf, n_trials=100)

best_n_estimators = study.best_params['n_estimators']
best_max_features = study.best_params['max_features']
best_max_depth = study.best_params['max_depth']
best_min_samples_split = study.best_params['min_samples_split']
best_min_samples_leaf = study.best_params['min_samples_leaf']

print("Best Hyperparameters for Random Forest Model:")
print(f"n_estimators: {best_n_estimators}")
print(f"max_features: {best_max_features}")
print(f"max_depth: {best_max_depth}")
print(f"min_samples_split: {best_min_samples_split}")
print(f"best_min_samples_leaf: {best_min_samples_leaf}")
'''
best_n_estimators = 461
best_max_features = 0.18509899425135484
best_max_depth = 36
best_min_samples_split = 6
best_min_samples_leaf = 10

rf_model = RandomForestRegressor(n_estimators=best_n_estimators, max_features=best_max_features, max_depth=best_max_depth,
                                 min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf, random_state=42)
rf_model.fit(x_train, y_train)

rf_model_pred = test[['season', 'round', 'driver', 'finishing_position']].copy()
rf_model_pred['Predicted'] = rf_model.predict(x_test)
rf_model_pred['Predicted Position'] = rf_model_pred.groupby(['season', 'round'])['Predicted'].rank()
rf_model_pred.to_excel('RF_model_pred.xlsx', index=False)

performance_metric(rf_model_pred, 'Random Forest Regressor')
print(model_performance)

# Support Vector Regressor
'''
def objective_svr(trial):
    params = {'gamma': trial.suggest_float('gamma', 1e-6, 1e-2, log=True), 'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])}
    svr_model = SVR(**params)
    scores = -cross_val_score(svr_model, x_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective_svr, n_trials=100)

best_gamma = study.best_params['gamma']
best_C = study.best_params['C']
best_kernel = study.best_params['kernel']

print("Best Hyperparameters for SVR Model:")
print(f"gamma: {best_gamma}")
print(f"C: {best_C}")
print(f"kernel: {best_kernel}")
'''
best_gamma = 2.9107472666424752e-05
best_C = 0.011470371755540337
best_kernel = 'linear'

svr_model = SVR(gamma=best_gamma, C=best_C, kernel=best_kernel)
svr_model.fit(x_train, y_train)
svr_model_pred = svr_model.predict(x_test)
svr_model_pred_df = test[['season', 'round', 'driver', 'finishing_position']].copy()
svr_model_pred_df['Predicted'] = svr_model_pred
svr_model_pred_df['Predicted Position'] = svr_model_pred_df.groupby(['season', 'round'])['Predicted'].rank()
svr_model_pred_df.to_excel('SVR_model_pred.xlsx', index=False)

performance_metric(svr_model_pred_df, 'Support Vector Regressor')
print(model_performance)

# NEURAL NETWORK

'''
def objective(trial):
    layer1_neurons = trial.suggest_int('layer1_neurons', 79, 81)
    layer2_neurons = trial.suggest_int('layer2_neurons', 19, 21)
    layer3_neurons = trial.suggest_int('layer3_neurons', 39, 41)
    layer4_neurons = trial.suggest_int('layer4_neurons', 4, 6)

    hidden_layer_sizes = (layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons)
    activation = trial.suggest_categorical('activation', ['relu'])
    solver = trial.suggest_categorical('solver', ['adam'])
    alpha = trial.suggest_float('alpha', 0.0001, 0.1, log=True)

    nn_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        max_iter=1000,
        random_state=42
    )

    scores = -cross_val_score(nn_model, x_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    return scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

best_hidden_layer_sizes = study.best_params['hidden_layer_sizes']
best_activation = study.best_params['activation']
best_solver = study.best_params['solver']
best_alpha = study.best_params['alpha']

print("Best Hyperparameters for Neural Network:")
print(f"hidden_layer_sizes: {best_hidden_layer_sizes}")
print(f"activation: {best_activation}")
print(f"solver: {best_solver}")
print(f"alpha: {best_alpha}")
'''
best_hidden_layer_sizes = (80, 20, 40, 5)
best_activation = 'relu'
best_solver = 'adam'
best_alpha = 0.0009

nn_model = MLPRegressor(
    hidden_layer_sizes=best_hidden_layer_sizes,
    activation=best_activation,
    solver=best_solver,
    alpha=best_alpha,
    max_iter=1000,
    random_state=42
)
nn_model.fit(x_train, y_train)

nn_model_pred = nn_model.predict(x_test)

nn_model_pred_df = test[['season', 'round', 'driver', 'finishing_position']].copy()
nn_model_pred_df['Predicted'] = nn_model_pred
nn_model_pred_df['Predicted Position'] = nn_model_pred_df.groupby(['season', 'round'])['Predicted'].rank()
nn_model_pred_df.to_excel('NN_model_pred.xlsx', index=False)

performance_metric(nn_model_pred_df, 'Neural Network')
print(model_performance)

model_performance.to_excel('Model_PerformanceV3a.xlsx', index=False)


