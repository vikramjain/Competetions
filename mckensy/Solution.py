import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from numpy import nan
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
test_IDs = test['ID']
train.Approved = pd.Categorical(train.Approved)
Y_train = train['Approved']
train.drop('Approved', axis=1, inplace=True)


def correctDataFrame(data):   
	data.replace('nan', nan, inplace=True)
	print(data.isnull().sum())
	data.Gender = pd.Categorical(data.Gender)
	data.Source = pd.Categorical(data.Source)
	data.Source_Category = pd.Categorical(data.Source_Category)
	data.City_Code = pd.Categorical(data.City_Code)
	data.City_Category = pd.Categorical(data.City_Category)
	data.Employer_Code = pd.Categorical(data.Employer_Code)
	data.Var1 = pd.Categorical(data.Var1)
	#impute City_Code
	impute_grps = data.pivot_table(values=['City_Code'], index =['Gender','Source','Source_Category'], aggfunc=lambda x: x.mode().iat[0])
	for i,row in data.loc[data['City_Code'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['Source'],row['Source_Category']])
		data.loc[i,'City_Code'] = impute_grps.loc[ind].values[0]
	#impute city_category
	impute_grps = data.pivot_table(values=['City_Category'], index =['City_Code','Gender'], aggfunc=lambda x: x.mode().iat[0])
	for i,row in data.loc[data['City_Category'].isnull(),:].iterrows():
		ind = tuple([row['City_Code'],row['Gender']])
		data.loc[i,'City_Category'] = impute_grps.loc[ind].values[0]
	
	
	#drops columns
	data.drop('ID', axis=1, inplace=True)
	data.drop('DOB', axis=1, inplace=True)
	data.drop('Lead_Creation_Date', axis=1, inplace=True)
	data.drop('Employer_Category2', axis=1, inplace=True)
	data.drop('Customer_Existing_Primary_Bank_Code', axis=1, inplace=True)
	data.drop('Primary_Bank_Type', axis=1, inplace=True)
	data.drop('Contacted', axis=1, inplace=True)
	
	data['Employer_Code'] = data['Employer_Code'].astype(str)
	data['Employer_Category1'] = data['Employer_Category1'].astype(str)
	#impute City_Code
	impute_grps = data.pivot_table(values=['Employer_Code'], index =['Gender','City_Code','City_Category'], aggfunc=lambda x: x.mode().iat[0])
	impute_grps_one = data.pivot_table(values=['Employer_Category1'], index =['Gender','City_Code','City_Category'], aggfunc=lambda x: x.mode().iat[0])
	data.Employer_Code = pd.Categorical(data.Employer_Code)
	data.Employer_Category1 = pd.Categorical(data.Employer_Category1)
	#iterate only through rows with missing City_Code
	for i,row in data.loc[data['Employer_Code'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['City_Code'],row['City_Category']])
		data.loc[i,'Employer_Code'] = impute_grps.loc[ind].values[0]
		data.loc[i,'Employer_Category1'] = impute_grps_one.loc[ind].values[0]
	
	impute_grps = data.pivot_table(values=['Existing_EMI'], index =['Var1', 'Employer_Category1'], aggfunc=np.mean)
	
	#iterate only through rows with missing LoanAmount
	for i,row in data.loc[data['Existing_EMI'].isnull(),:].iterrows():
		ind = tuple([row['Var1'],row['Employer_Category1']])
		data.loc[i,'Existing_EMI'] = impute_grps.loc[ind].values[0]
	
	impute_grps = data.pivot_table(values=['Loan_Amount'], index =['Gender','Employer_Category1'], aggfunc=np.mean)
	for i,row in data.loc[data['Loan_Amount'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['Employer_Category1']])
		data.loc[i,'Loan_Amount'] = impute_grps.loc[ind].values[0]
	
	impute_grps = data.pivot_table(values=['Loan_Period'], index =['Gender','Employer_Category1'], aggfunc=np.mean)
	for i,row in data.loc[data['Loan_Period'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['Employer_Category1']])
		data.loc[i,'Loan_Period'] = impute_grps.loc[ind].values[0]
		
	impute_grps = data.pivot_table(values=['Interest_Rate'], index =['Gender','Employer_Category1'], aggfunc=np.mean)
	#iterate only through rows with missing LoanAmount
	for i,row in data.loc[data['Interest_Rate'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['Employer_Category1']])
		data.loc[i,'Interest_Rate'] = impute_grps.loc[ind].values[0]
		
	impute_grps = data.pivot_table(values=['EMI'], index =['Gender','Employer_Category1'], aggfunc=np.mean)
	#iterate only through rows with missing LoanAmount
	for i,row in data.loc[data['EMI'].isnull(),:].iterrows():
		ind = tuple([row['Gender'],row['Employer_Category1']])
		data.loc[i,'EMI'] = impute_grps.loc[ind].values[0]
	

	#data.drop('City_Code', axis=1, inplace=True)
	#data.drop('City_Category', axis=1, inplace=True)
	#data.drop('Employer_Code', axis=1, inplace=True)
	#data.drop('Employer_Category1', axis=1, inplace=True)
	#data.drop('Source', axis=1, inplace=True)
	#data.drop('Source_Category', axis=1, inplace=True)
	#data.drop('Gender', axis=1, inplace=True)
	#data.drop('Loan_Period', axis=1, inplace=True)
	#data.drop('Interest_Rate', axis=1, inplace=True)
	le = LabelEncoder()
	
	train_vals = list(data.City_Code.unique())
	le.fit(train_vals)
	data.City_Code = le.transform(data.City_Code)
	
	train_vals = list(data.City_Category.unique())
	le.fit(train_vals)
	data.City_Category = le.transform(data.City_Category)
	
	train_vals = list(data.Employer_Code.unique())
	le.fit(train_vals)
	data.Employer_Code = le.transform(data.Employer_Code)
	
	train_vals = list(data.Source.unique())
	le.fit(train_vals)
	data.Source = le.transform(data.Source)
	
	train_vals = list(data.Source_Category.unique())
	le.fit(train_vals)
	data.Source_Category = le.transform(data.Source_Category)
	
	train_vals = list(data.Gender.unique())
	le.fit(train_vals)
	data.Gender = le.transform(data.Gender)
	
	train_vals = list(data.Employer_Category1.unique())
	le.fit(train_vals)
	data.Employer_Category1 = le.transform(data.Employer_Category1)
	
	data['Existing_EMI'] = data['Existing_EMI'].astype(float)
	data['Loan_Amount'] = data['Loan_Amount'].astype(float)
	data['Loan_Period'] = data['Loan_Period'].astype(float)
	data['Interest_Rate'] = data['Interest_Rate'].astype(float)
	data['EMI'] = data['EMI'].astype(float)
	
	print(data.head())
	print(data.isnull().sum())
	print(data.dtypes)
	return data

train = correctDataFrame(train)
test = correctDataFrame(test)

X_train, X_valid, y_train, y_valid = train_test_split(train, Y_train, test_size=0.1, random_state = 12)

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


print('Training LGBM model...')
params = {}
params['learning_rate'] = 0.1
params['application'] = 'binary'
params['max_depth'] = 10
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'application': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 16,
    'learning_rate': 0.1,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'min_data': 50,	
    'max_depth': 10,	
    'verbose': 0
}
'''
params = {'task': 'train',
    'boosting_type': 'gbdt',
    'application': 'binary',
    'metric': {'l1', 'auc'},
    'learning_rate': 0.1,
    'max_depth': 7,
    'num_leaves': 17,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.6,
    'bagging_freq': 17}
	'''
model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, \
early_stopping_rounds=10, verbose_eval=10)

print('Making predictions and saving them...')
p_test = model.predict(test)

subm = pd.DataFrame()
subm['ID'] = test_IDs
subm['Approved'] = p_test
subm.to_csv('result.csv', index=False, sep=',')
print('Done!')

'''
logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])


X_train, X_test, y_train, y_test = train_test_split(train, Y_train, random_state=0, test_size=0.25)

pca.fit(X_train)

n_components = [2, 3, 4]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,dict(pca__n_components=n_components,logistic__C=Cs))
estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(estimator.score(X_test, y_test)))

print(classification_report(y_test, predictions))
print(predictions)

predictions = estimator.predict(test)
print(predictions)

df = pd.DataFrame(test_IDs)
result = pd.DataFrame(predictions)
df = df.join(result)
df.columns = ['ID', 'Approved']
df.to_csv('result.csv', index=False, sep=',')
'''
'''
X_train = train
print(X_train.shape)
Y_train = Y_train
X_test  = test

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
parameter_grid = {
                 'max_depth' : [4,5],
                 'n_estimators': [50,100],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(Y_train , n_folds=5)
random_forest = GridSearchCV(random_forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

random_forest.fit(X_train, Y_train)

predictions = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))

df = pd.DataFrame(test_IDs)
result = pd.DataFrame(predictions)
df = df.join(result)
df.columns = ['ID', 'Approved']
df.to_csv('result.csv', index=False, sep=',')
'''