import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('final_df.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

train = df[df.season < 2021]
test = df[df['season'] > 2020]

x_train = train.drop(['driver' , 'finishing_position'], axis = 1)
y_train = train.finishing_position
x_test = test.drop(['driver', 'finishing_position'], axis=1)
y_test = test.finishing_position

scaler = MinMaxScaler()
driver_age = df['driver_age'].values.reshape(-1, 1)
scaled_driver_age = scaler.fit_transform(driver_age)
df['driver_age'] = scaled_driver_age

tscv = TimeSeriesSplit(n_splits = 10)
weights = [0, 0.25, 0.5, 0.75, 1]
model_performance = pd.DataFrame(columns=['Model', 'Weight', 'Performance', 'Percentage Correct positions','Percentage Correct Wins',
                                         'Percentage Correct 2nd Place', 'Percentage Correct 3rd Place'])


def performance_metric(df, weight, model_name):
    global model_performance
    y_true = df['finishing_position']
    y_pred = df['Predicted Position']

    mae = mean_absolute_error(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    perf_metric = (weight * spearman) + ((1 - weight) * mae)
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

    model_performance.loc[len(model_performance)] = {'Model': model_name, 'Weight': weight,
                                                     'Performance': perf_metric,
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

for weight in weights:
    performance_metric(lin_reg_model_pred, weight, 'Linear Regression')

#print(model_performance)

# DECISION TREE MODEL (USELESS - Linear Reg way better.)

'''
params_dt = {'max_depth': range(3, 51, 5), 'min_samples_leaf': range(2, 16)}
grid_search_dt = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=params_dt, cv=tscv)
grid_search_dt.fit(x_train, y_train)
best_max_depth = grid_search_dt.best_params_['max_depth']
best_min_samples_leaf = grid_search_dt.best_params_['min_samples_leaf']

print("Best Hyperparameters for Decision Tree Model:")
print(f"Max Depth: {best_max_depth}")
print(f"Min Samples Leaf: {best_min_samples_leaf}")
'''
# Gridsearched Best hyperparameters:
best_max_depth = 4
best_min_samples_leaf = 14

dt_model = DecisionTreeRegressor(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
dt_model.fit(x_train, y_train)
dt_model_pred = test[['season', 'round', 'driver','finishing_position']].copy()
dt_model_pred['Predicted'] = dt_model.predict(x_test)
dt_model_pred['Predicted Position'] = dt_model_pred.groupby(['season', 'round'])['Predicted'].rank()
dt_model_pred.to_excel('DT_model_pred.xlsx', index=False)

for weight in weights:
    performance_metric(dt_model_pred, weight, 'Decision Tree')

#print(model_performance)

# RANDOM FOREST MODEL
'''
params_rf = {'max_features': [1.0 , 'sqrt', 'log2'], 'max_depth': range(3, 51, 5), 'n_estimators': [50, 100]}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=params_rf, cv=tscv)
grid_search_rf.fit(x_train, y_train)

best_n_estimators = grid_search_rf.best_params_['n_estimators']
best_max_features = grid_search_rf.best_params_['max_features']
best_max_depth = grid_search_rf.best_params_['max_depth']

print("Best Hyperparameters for Random Forest Model:")
print(f"n_estimators: {best_n_estimators}")
print(f"max_features: {best_max_features}")
print(f"max_depth: {best_max_depth}")
'''
# Gridsearched Best hyperparameters:
best_n_estimators = 180
best_max_features = 0.5
best_max_depth = 6

rf_model = RandomForestRegressor(n_estimators=best_n_estimators, max_features=best_max_features, max_depth=best_max_depth, random_state=42)
rf_model.fit(x_train, y_train)

rf_model_pred = test[['season', 'round', 'driver', 'finishing_position']].copy()
rf_model_pred['Predicted'] = rf_model.predict(x_test)
rf_model_pred['Predicted Position'] = rf_model_pred.groupby(['season', 'round'])['Predicted'].rank()
rf_model_pred.to_excel('RF_model_pred.xlsx', index=False)

for weight in weights:
    performance_metric(rf_model_pred, weight, 'Random Forest Regressor')

#print(model_performance)

# Support Vector Regressor
'''
params_svr = {'gamma': [0.000001, 0.00001, 0.00005], 'C': [0.0008, 0.001, 0.002], 'kernel': ['linear']}
grid_search_svr = GridSearchCV(estimator=SVR(), param_grid=params_svr, cv=tscv, verbose= 200)
grid_search_svr.fit(x_train, y_train)

best_gamma = grid_search_svr.best_params_['gamma']
best_C = grid_search_svr.best_params_['C']
best_kernel = grid_search_svr.best_params_['kernel']

print("Best Hyperparameters for SVR Model:")
print(f"gamma: {best_gamma}")
print(f"C: {best_C}")
print(f"kernel: {best_kernel}")
'''
# Gridsearched Best hyperparameters:
best_gamma = 0.000001
best_C = 0.001
best_kernel = 'linear'

svr_model = SVR(gamma=best_gamma, C=best_C, kernel=best_kernel)
svr_model.fit(x_train, y_train)
svr_model_pred = svr_model.predict(x_test)
svr_model_pred_df = test[['season', 'round', 'driver', 'finishing_position']].copy()
svr_model_pred_df['Predicted'] = svr_model_pred
svr_model_pred_df['Predicted Position'] = svr_model_pred_df.groupby(['season', 'round'])['Predicted'].rank()
svr_model_pred_df.to_excel('SVR_model_pred.xlsx', index=False)

for weight in weights:
    performance_metric(svr_model_pred_df, weight, 'Support Vector Regressor')

#print(model_performance)

# Neural Network
'''
params_nn = {'hidden_layer_sizes': [(80, 20, 40, 5),(80, 30, 5), (80, 40, 10, 5)],
    'activation': ['relu', 'sigmoid', 'tanh'], 'solver': ['adam'],
    'alpha': [0.0009, 0.00095, 0.00085]}

grid_search_nn = GridSearchCV(estimator=MLPRegressor(random_state=42), param_grid=params_nn, cv=tscv, verbose = 200)
grid_search_nn.fit(x_train, y_train)

best_hidden_layer_sizes = grid_search_nn.best_params_['hidden_layer_sizes']
best_activation = grid_search_nn.best_params_['activation']
best_solver = grid_search_nn.best_params_['solver']
best_alpha = grid_search_nn.best_params_['alpha']

print("Best Hyperparameters for Neural Network:")
print(f"hidden_layer_sizes: {best_hidden_layer_sizes}")
print(f"activation: {best_activation}")
print(f"solver: {best_solver}")
print(f"alpha: {best_alpha}")
'''
# Gridsearched Best hyperparameters:
best_hidden_layer_sizes = (80, 20, 40, 5)
best_activation = 'relu'
best_solver = 'adam'
best_alpha = 0.0009

nn_model = MLPRegressor(hidden_layer_sizes=best_hidden_layer_sizes, activation=best_activation,
                        solver=best_solver, alpha=best_alpha, random_state=42)
nn_model.fit(x_train, y_train)

nn_model_pred = nn_model.predict(x_test)

nn_model_pred_df = test[['season', 'round', 'driver', 'finishing_position']].copy()
nn_model_pred_df['Predicted'] = nn_model_pred
nn_model_pred_df['Predicted Position'] = nn_model_pred_df.groupby(['season', 'round'])['Predicted'].rank()
nn_model_pred_df.to_excel('NN_model_pred.xlsx', index=False)

for weight in weights:
    performance_metric(nn_model_pred_df, weight, 'Neural Network')

model_performance.to_excel('Model_PerformanceV2.xlsx', index=False)
print(model_performance)
