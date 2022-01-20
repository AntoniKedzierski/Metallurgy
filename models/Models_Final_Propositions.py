# IMPORTING TOOLS
install('autokeras')
install("xgboost")
install("sklearn")
from numpy import mean
from numpy import std
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from autokeras import StructuredDataRegressor


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# SELECTING VARIABLES
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

parameters_svr = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.6, 0.9],'C': [10, 100, 1000, 10000]}]
parameters_xgboost = [{    'max_depth':[3,7,10],
                            'min_child_weight':[1,2,5],
                            'gamma': [0, 0.01, 0.1],
                            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                            "learning_rate" :[0.01,1,100]
                            }]
parameters_lasso = [{'alpha': [0,0.01,0.1,1,10,50] }]
parameter_MLPC = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,50), (150,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

svr=GridSearchCV(estimator=SVR(), param_distributions = parameters_svr, cv=cv, scoring='neg_mean_squared_error' )
xgbr=RandomizedSearchCV(estimator=XGBRegressor(), param_distributions = parameters_xgboost , cv=cv, scoring='neg_mean_squared_error' )
lasso=GridSearchCV(estimator=Lasso(), param_distributions = parameters_lasso, cv=cv, scoring='neg_mean_squared_error' )
mlpc=RandomizedSearchCV(estimator=MLPC(), param_distributions = parameters_MLPC , cv=cv, scoring='neg_mean_squared_error' )
AutoKeras=StructuredDataRegressor(max_trials=100, loss='mean_absolute_error')

svr.fit()
xgbr.fit()
lasso.fit()
mlpc.fit()
AutoKeras.fit()