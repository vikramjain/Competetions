import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_df1 = pd.read_csv("test.csv")

train_df['Item_Outlet_Sales_Number'] = train_df['Item_Outlet_Sales']/train_df['Item_MRP']
# convert from float to int
print(train_df.shape)
print(test_df.shape)
y_Item_Outlet_Sales = pd.DataFrame(train_df.Item_Outlet_Sales.copy())
y_Item_Outlet_Sales.columns = ['Item_Outlet_Sales']
y_Item_Outlet_Sales_Number = train_df.Item_Outlet_Sales_Number.copy()
train_df.drop('Item_Outlet_Sales', axis=1, inplace=True)
train_df.drop('Item_Outlet_Sales_Number', axis=1, inplace=True)
df = pd.concat([train_df, test_df], axis=0)
print('test shape')
print(y_Item_Outlet_Sales.shape)
print(df.head())
print(df.shape)
print(df.isnull().sum())
df = df.dropna(how='all')
r,c = df.shape


##################################Now treatment of columns is being done #####################################################
#Item_Fat_Content
print(df['Item_Fat_Content'].unique())
df['Item_Fat_Content'] = df['Item_Fat_Content'].str.lower()
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['lf'], 'low fat')
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(['reg'], 'regular')
print(df['Item_Fat_Content'].unique())
df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'low fat' : 0, 'regular' : 1})
print(df['Item_Fat_Content'].unique())

#Outlet_Location_Type
print(df['Outlet_Location_Type'].unique())
df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map({'Tier 1' : 1, 'Tier 2' : 2, 'Tier 3' : 3})
print(df['Outlet_Location_Type'].unique())

#Outlet_Type
print(df['Outlet_Type'].unique())
df['Outlet_Type'] = df['Outlet_Type'].map({'Supermarket Type1' : 1, 'Supermarket Type2' : 2, 'Supermarket Type3' : 3, 'Grocery Store' : 4})
print(df['Outlet_Type'].unique())

#Outlet_Type
print(df['Item_Type'].unique())
df['Item_Type'] = df['Item_Type'].map({'Dairy' : 1, 'Soft Drinks' : 2, 'Meat' : 3, 'Fruits and Vegetables' : 4, 'Household' : 5, 'Baking Goods' : 6, 'Snack Foods' : 7, 'Frozen Foods' : 8, 'Breakfast' : 9, 'Health and Hygiene' : 10, 'Hard Drinks': 11, 'Canned' : 12, 'Breads' : 13, 'Starchy Foods' : 14, 'Others' : 15,'Seafood' : 16})
print(df['Item_Type'].unique())

print(df.head())

# replace nan on Item weight and Outlet size
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
df['Outlet_Size'] = df['Outlet_Size'].map({'High' : 1, 'Medium' : 2, 'Small' : 3})
print(df['Outlet_Size'].unique())

#df.drop('Item_Visibility', axis=1, inplace=True)

#df.hist(column='Item_MRP', bins=50)

figure = plt.figure(figsize=(8,5))
plt.hist(df['Item_MRP'], bins=np.arange(df['Item_MRP'].min(), df['Item_MRP'].max()+1))
plt.xlabel('Item_MRP')
plt.legend()
plt.show()

df['Item_Type'] = (df['Item_Type']).astype(int)
df.loc[ df['Item_MRP'] <= 69, 'Item_MRP'] = 1
df.loc[(df['Item_MRP'] > 69) & (df['Item_MRP'] <= 136), 'Item_MRP'] = 2
df.loc[(df['Item_MRP'] > 136) & (df['Item_MRP'] <= 203), 'Item_MRP'] = 3
df.loc[(df['Item_MRP'] > 203), 'Item_MRP'] = 4


max_visi = df['Item_Visibility'].max()
min_visi = df['Item_Visibility'].min()
qrt_visi = (max_visi + min_visi)/4 * 1000
print(qrt_visi)
df['Item_Visibility'] = df['Item_Visibility'].astype(int)
df["Item_Visibility"] = df["Item_Visibility"].apply(lambda x: x * 1000)
df.loc[ df['Item_Visibility'] <= qrt_visi, 'Item_Visibility'] = 1
df.loc[(df['Item_Visibility'] > qrt_visi) & (df['Item_Visibility'] <= (2*qrt_visi)), 'Item_MRP'] = 2
df.loc[(df['Item_Visibility'] > (qrt_visi*2)) & (df['Item_Visibility'] <= (3*qrt_visi)), 'Item_MRP'] = 3
df.loc[(df['Item_MRP'] > (qrt_visi*3)), 'Item_MRP'] = 4
print(df['Item_MRP'].unique())

df["Outlet_Establishment_Year"] = df["Outlet_Establishment_Year"].apply(lambda x: 2013 - x)


		
print(df.shape)
Item_Fat_Content_dummies = pd.get_dummies(df['Item_Fat_Content'],prefix="Item_Fat_Content",drop_first=True)
# adding dummy variables
df = pd.concat([df,Item_Fat_Content_dummies],axis=1)    
df.drop('Item_Fat_Content',axis=1,inplace=True)

Item_Visibility_dummies = pd.get_dummies(df['Item_Visibility'],prefix="Item_Visibility",drop_first=True)
# adding dummy variables
df = pd.concat([df,Item_Visibility_dummies],axis=1)    
df.drop('Item_Visibility',axis=1,inplace=True)

Item_Type_dummies = pd.get_dummies(df['Item_Type'],prefix="Item_Type",drop_first=True)
# adding dummy variables
df = pd.concat([df,Item_Type_dummies],axis=1)    
df.drop('Item_Type',axis=1,inplace=True)

Item_MRP_dummies = pd.get_dummies(df['Item_MRP'],prefix="Item_MRP",drop_first=True)
# adding dummy variables
df = pd.concat([df,Item_MRP_dummies],axis=1)    
df.drop('Item_MRP',axis=1,inplace=True)

Outlet_Size_dummies = pd.get_dummies(df['Outlet_Size'],prefix="Outlet_Size",drop_first=True)
# adding dummy variables
df = pd.concat([df,Outlet_Size_dummies],axis=1)    
df.drop('Outlet_Size',axis=1,inplace=True)

Outlet_Location_Type_dummies = pd.get_dummies(df['Outlet_Location_Type'],prefix="Outlet_Location_Type",drop_first=True)
# adding dummy variables
df = pd.concat([df,Outlet_Location_Type_dummies],axis=1)    
df.drop('Outlet_Location_Type',axis=1,inplace=True)

Outlet_Type_dummies = pd.get_dummies(df['Outlet_Type'],prefix="Outlet_Type",drop_first=True)
# adding dummy variables
df = pd.concat([df,Outlet_Type_dummies],axis=1)    
df.drop('Outlet_Type',axis=1,inplace=True)

df.drop('Item_Identifier',axis=1,inplace=True)
df.drop('Outlet_Identifier',axis=1,inplace=True)

print(df.isnull().sum())

train = df.iloc[:8523,]
test = df.iloc[8523:,:]
print(train.shape)
print(y_Item_Outlet_Sales.shape)


#train['Item_Outlet_Sales'] = y_Item_Outlet_Sales[['Item_Outlet_Sales']]
estimators = [50, 75, 100, 125, 150, 200, 250, 500]
'''
for e in estimators:
    xg = xgb.XGBRegressor(n_estimators = e)
    
#     plt.figure(figsize=(10,10))
    xg.fit(train, y_Item_Outlet_Sales['Item_Outlet_Sales'])
    print(train.dtypes)	
    print(xg.feature_importances_)
    xgb.plot_importance(xg, importance_type='gain')#, ax=ax1)
'''	
def prediction_function(train, test):
    estimators = [50, 75, 80, 100, 125, 150, 200, 250, 500]

    final = []

    for e in estimators:
        rf = RandomForestRegressor(random_state = 1, n_estimators = e, min_samples_split = 8, min_samples_leaf = 4)
        gbr = GradientBoostingRegressor(random_state = 1, n_estimators = e, min_samples_split = 8, 
                                        min_samples_leaf = 4, learning_rate = 0.1)
        xg = xgb.XGBRegressor(n_estimators = e)

        rf.fit(train, train['Item_Outlet_Sales'])
        predictions_rf = rf.predict(test)
        predictions_rf = predictions_rf.astype(int)

        gbr.fit(train, train['Item_Outlet_Sales'])
        predictions_gbr = gbr.predict(test)
        predictions_gbr = predictions_gbr.astype(int)

        xg.fit(train, train['Item_Outlet_Sales'])
        predictions_xg = xg.predict(test)
        predictions_xg = predictions_xg.astype(int)

        mse_rf = (np.sqrt(mean_squared_error(test['Item_Outlet_Sales'], predictions_rf)), 'RF')
        mse_gbr = (np.sqrt(mean_squared_error(test['Item_Outlet_Sales'], predictions_gbr)), 'GBR')
        mse_xg = (np.sqrt(mean_squared_error(test['Item_Outlet_Sales'], predictions_xg)), 'XGB')

        error_min = min(mse_rf, min(mse_gbr, mse_xg))
#         print(error_min)
        final.append((error_min, e))

    min_final = min(final)
    print("Minimum MSE, regressor to use and number of estimators: "+str(min_final))
    return list(min_final)
train = pd.concat([train, y_Item_Outlet_Sales], axis=1)
#print(train.head())
	
train_tmp = train[:int(train.shape[0]*0.8)]
test_tmp = train[int(train.shape[0]*0.8):]

min_final = prediction_function(train_tmp, test_tmp)

e_to_use = min_final[1]
regressor_to_use = min_final[0][1]
# print("Mimimum RMSE error was for "+str(regressor_to_use)+" with "+str(e_to_use)+" estimators")

if(regressor_to_use == 'RF'):
    reg = RandomForestRegressor(random_state = 1, n_estimators = e_to_use, min_samples_split = 8, min_samples_leaf = 4)
elif(regressor_to_use == 'GBR'):
    reg = GradientBoostingRegressor(random_state = 1, n_estimators = e_to_use, min_samples_split = 8, min_samples_leaf = 4, learning_rate = 0.2)
else:
    reg = xgb.XGBRegressor(n_estimators = e_to_use)
train.drop('Item_Outlet_Sales', axis=1, inplace=True)
reg.fit(train, y_Item_Outlet_Sales['Item_Outlet_Sales'])
predictions = reg.predict(test)
predictions = predictions.astype(int)

sub = pd.DataFrame({'Item_Identifier' : test_df1['Item_Identifier'], 'Outlet_Identifier' : test_df1['Outlet_Identifier'], 'Item_Outlet_Sales' : predictions})
sub.to_csv('output.csv', index=False)