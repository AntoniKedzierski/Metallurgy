# IMPORTING TOOLS
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# SELECTING VARIABLES
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.6, 0.9],'C': [10, 100, 1000, 10000]}]
GridSearchCV(estimator=SVR, param_distributions = parameters, cv=cv )


#cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')