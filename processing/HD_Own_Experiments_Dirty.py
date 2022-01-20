# ctrl + shift + F10
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#!pip install tpot
#!pip install statsmodels
print ("AA")
# TODO:
# 1. EDA
# 1.1 Pandas profiling
# 1.2 Missingness
# 1.3 Dedicated datasets and missingness
# 1.4 Colinearity in datastes
# 1.5 Outlier detection
# 1.6 Visualization of characteristics
# 1.7 Tests

# 2. Preprocessing
# Scaling
# Outlier elimination
# Different versions of strategies


# 3. Prognozing the missingness
# 3.1 TPOT for 1 miss columns
# 3.2 Custom Probabilistic Search algorithms for 1 miss colmn
# 3.3 Validation of models selected for 1 miss column
# 3.4 General missingness algorithms
# 3.5 Deep neural net for misingness

# 4. Prognozing the variable of interest
#
#

import numpy as np
import pandas as pd
import tpot as tpot
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Data for AS CAST
DF=pd.read_excel("C:\\Users\\huber\\Dropbox\\PW\\3 SEMESTR\\ODLEWNICTWO\\dane_zeliwo_full.xlsx")
DF=DF.drop(['Lp', '[LABELS]', 'tward'], axis=1)
DF.head()
DF.iloc[:,0:6].describe()
DF.iloc[:,6:11].describe()



# calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = DF.columns
vif_data["VIF"] = [variance_inflation_factor(DF.values, i)
                   for i in range(len(DF.columns))]
# KRZEM TWARD WYTRZYM WEGIEL są bardzo dobrze tłumaczone przez pozostały zestaw zmiennych
vif_data["feature"] = DF.drop(['tward','wydluz', 'wytrzym', 'wegiel'], axis=1).columns
vif_data["VIF"] = [variance_inflation_factor(DF.drop(['tward','wydluz', 'wytrzym', 'wegiel'], axis=1).values, i)
                   for i in range(len(DF.drop(['tward','wydluz', 'wytrzym', 'wegiel'], axis=1).columns))]

# Full data model tpot for wytrzym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# Average CV score on the training set was: -1101.1522789010987
wytrzym_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    Normalizer(norm="max"),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
)
# Full data model tpot for wydluz
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# Average CV score on the training set was: -4.896314428021981
wydluz_pipeline = make_pipeline(
    Normalizer(norm="l1"),
    StackingEstimator(estimator=LinearSVR(C=20.0, dual=True, epsilon=0.1, loss="epsilon_insensitive", tol=0.01)),
    Normalizer(norm="max"),
    ExtraTreesRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
)
# Learning models:
DF=pd.read_excel("C:\\Users\\huber\\Dropbox\\PW\\3 SEMESTR\\ODLEWNICTWO\\dane_zeliwo_full.xlsx")
DF=DF.drop(['Lp', '[LABELS]'], axis=1)
X_train=DF.iloc[:,0:9]
wytrzym_pipeline.fit(DF.iloc[:,0:9],DF.wytrzym)
wydluz_pipeline.fit(DF.iloc[:,0:9], DF.wydluz)


import pickle
pickle.dump(wydluz_pipeline, open('models/imputation/wytrzym_ascast.sav', 'wb'))
loaded_model = pickle.load(open('models/imputation/wytrzym_ascast.sav', 'rb'))

pickle.dump(wytrzym_pipeline, open('models/imputation/wydluz_ascast.sav', 'wb'))
loaded_model = pickle.load(open('models/imputation/wydluz_ascast.sav', 'rb'))

loaded_model.predict(DF.iloc[:,0:9])




# Data model without wegiel tpot for wytrzym

# Data model without wegiel tpot for wydluz

# Private models :
 # Regression in R Lasso

 # SVMR

 # GLR

 # Ensambe Voting Regressor

 # MLP Regressor





# import tpot
# from tpot import TPOTRegressor
loaded_model = pickle.load(open('wytrzym_ascast.sav', 'rb'))  # wczytany zapisany model
DF=pd.read_excel("C:\\Users\\huber\\Dropbox\\PW\\3 SEMESTR\\ODLEWNICTWO\\dane_0.xlsx")

