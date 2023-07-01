import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('/content/drive/MyDrive/final_df.csv')
df = data.copy()
df.rename(columns={'podium': 'finishing_position'}, inplace=True)

grouped = df.groupby(['season', 'round'])
highest_values = grouped[['driver_points', 'constructor_points']].transform('max')
df['driver_points_percentage'] = df['driver_points'] / highest_values['driver_points'] 
df['constructor_points_percentage'] = df['constructor_points'] / highest_values['constructor_points'] 
df['driver_points_percentage'] = df['driver_points_percentage'].fillna(0)
df['constructor_points_percentage'] = df['constructor_points_percentage'].fillna(0)
columns_to_drop = ['driver_points', 'constructor_points']
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

# RANDOM FOREST MODEL

params_rf = {'n_estimators': [50, 100, 200, 300, 400], 'max_features': [1.0, 'sqrt', 'log2', 0.8, 0.5],
    'max_depth': [3, 5, 10, 20, 30, 55], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=params_rf, cv=tscv, verbose = 200)
grid_search_rf.fit(x_train, y_train)

best_n_estimators = grid_search_rf.best_params_['n_estimators']
best_max_features = grid_search_rf.best_params_['max_features']
best_max_depth = grid_search_rf.best_params_['max_depth']

print("Best Hyperparameters for Random Forest Model:")
print(f"n_estimators: {best_n_estimators}")
print(f"max_features: {best_max_features}")
print(f"max_depth: {best_max_depth}")

rf_model = RandomForestRegressor(n_estimators=best_n_estimators, max_features=best_max_features, max_depth=best_max_depth, random_state=42)
rf_model.fit(x_train, y_train)

rf_model_pred = test[['season', 'round', 'driver', 'finishing_position']].copy()
rf_model_pred['Predicted'] = rf_model.predict(x_test)
rf_model_pred['Predicted Position'] = rf_model_pred.groupby(['season', 'round'])['Predicted'].rank()
rf_model_pred.to_excel('RF_model_pred.xlsx', index=False)

for weight in weights:
    performance_metric(rf_model_pred, weight, 'Random Forest Regressor')

model_performance.to_excel('Model_PerformanceV3b.xlsx', index=False)
print(model_performance)
