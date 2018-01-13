import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from numpy import nan
from sklearn.cross_validation import train_test_split
from pandas import Series,DataFrame
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

def get_title(name):
    if 'A' in name:
        return 1
    elif 'B' in name:
        return 2
    elif 'C' in name:
        return 2
		
train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
train.replace('nan', nan, inplace=True)
test.replace('nan', nan, inplace=True)
train.var2 = pd.Categorical(train.var2)
test.var2 = pd.Categorical(test.var2)
train['var2'] = train['var2'].apply(get_title) 
test['var2'] = test['var2'].apply(get_title) 
train.var2 = pd.Categorical(train.var2)
test.var2 = pd.Categorical(test.var2)

print(train.head())
print(train.isnull().sum())
print(train.dtypes)

# Cabin
train_Y = train["electricity_consumption"] 
train.drop("ID",axis=1,inplace=True)
train.drop("datetime",axis=1,inplace=True)
train.drop("electricity_consumption",axis=1,inplace=True)

test_id = test["ID"] 
test.drop("ID",axis=1,inplace=True)
test.drop("datetime",axis=1,inplace=True)

X_train = train
Y_train = train_Y
X_test = test

parameter_grid ={"n_estimators" : [50, 75, 100, 125, 150], "max_features": ["auto", "sqrt", "log2"], "min_samples_split" : [2,4,8], "bootstrap": [True, False]}

# Random Forests
random_forest = RandomForestRegressor(n_estimators=100)

random_forest = RandomForestRegressor(random_state = 1, n_estimators = 100, min_samples_split = 8, min_samples_leaf = 4)
cross_validation = StratifiedKFold(train_Y, n_folds=3)
random_forest = GridSearchCV(random_forest, param_grid=parameter_grid, cv=cross_validation)
'''	
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [50,100,120,150],
                 'criterion': ['gini','entropy'],
                 'max_features': ['auto', 'log2', 'sqrt', None],
                 'min_samples_split': [3,4,5,6,7]				 
                 }

cross_validation = StratifiedKFold(Y_train, n_folds=5)
random_forest = GridSearchCV(random_forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
'''
random_forest.fit(X_train, Y_train)

Y_pred_1 = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))
Y_pred = Y_pred_1
print(Y_pred)

submission = pd.DataFrame({
        "ID": test_id,
        "electricity_consumption": Y_pred
    })
submission.to_csv('final.csv', index=False)
