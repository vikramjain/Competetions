import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from numpy import nan
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('train.csv', sep=',')
data.replace('nan', nan, inplace=True)

data['Gender'].fillna('Male', inplace=True)
data['Married'].fillna('Yes', inplace=True)
data['Self_Employed'].fillna('No', inplace=True)
impute_grps = data.pivot_table(values=['LoanAmount'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]
impute_grps = data.pivot_table(values=['Loan_Amount_Term'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Loan_Amount_Term'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'Loan_Amount_Term'] = impute_grps.loc[ind].values[0]
impute_grps = data.pivot_table(values=['Dependents'], index =['Gender','Married','Education','Self_Employed'], aggfunc=lambda x: x.mode().iat[0])
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Dependents'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed']])
  data.loc[i,'Dependents'] = impute_grps.loc[ind].values[0]
  
impute_grps = data.pivot_table(values=['Credit_History'], index =['Gender','Married','Education','Self_Employed','Dependents'], aggfunc=lambda x: x.mode().iat[0])
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Credit_History'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed'],row['Dependents']])
  data.loc[i,'Credit_History'] = impute_grps.loc[ind].values[0]

def percConvert(ser):
  return ser/float(ser[-1])

print(pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True).apply(percConvert, axis=1))
data.Gender = pd.Categorical(data.Gender)
data.Married = pd.Categorical(data.Married)
data.Dependents = pd.Categorical(data.Dependents)
data.Education = pd.Categorical(data.Education)
data.Self_Employed = pd.Categorical(data.Self_Employed)
data.Property_Area = pd.Categorical(data.Property_Area)
data.Loan_Status = pd.Categorical(data.Loan_Status)

data_new = data.Loan_ID
data.drop('Loan_ID', axis=1, inplace=True)
for y in data.columns:
	if(data[y].dtypes.name == 'category'):
		print(data[y].name)
		print (data[y].unique())

df_Gender = pd.get_dummies(data['Gender'],prefix='Gender',drop_first=True)
df_Married = pd.get_dummies(data['Married'],prefix='Married',drop_first=True)
df_Dependents = pd.get_dummies(data['Dependents'],prefix='Dependents',drop_first=True)
df_Education = pd.get_dummies(data['Education'],prefix='Education',drop_first=True)
df_Self_Employed = pd.get_dummies(data['Self_Employed'],prefix='Self_Employed',drop_first=True)
df_Property_Area = pd.get_dummies(data['Property_Area'],prefix='Property_Area',drop_first=True)

data = data.join(df_Gender)
data = data.join(df_Married)
data = data.join(df_Dependents)
data = data.join(df_Education)
data = data.join(df_Self_Employed)
df = data.join(df_Property_Area)

df.drop('Gender', axis=1, inplace=True)
df.drop('Married', axis=1, inplace=True)
df.drop('Dependents', axis=1, inplace=True)
df.drop('Education', axis=1, inplace=True)
df.drop('Self_Employed', axis=1, inplace=True)
df.drop('Property_Area', axis=1, inplace=True)

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()

y_digits = df.Loan_Status
df.drop('Loan_Status', axis=1, inplace=True)
X_digits = df
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=0, test_size=0.25)

'''
X_digits = X_train
y_digits = y_train
# Plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()
# Prediction
n_components = [2, 10, 14]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_digits, y_digits)
predictions = estimator.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(estimator.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
print(predictions)
'''
'''
data_test = pd.read_csv('test.csv', sep=',')

data_test.replace('nan', nan, inplace=True)

data_test['Gender'].fillna('Male', inplace=True)
data_test['Married'].fillna('Yes', inplace=True)
data_test['Self_Employed'].fillna('No', inplace=True)
data = data_test
impute_grps = data.pivot_table(values=['LoanAmount'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]
impute_grps = data.pivot_table(values=['Loan_Amount_Term'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Loan_Amount_Term'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'Loan_Amount_Term'] = impute_grps.loc[ind].values[0]
impute_grps = data.pivot_table(values=['Dependents'], index =['Gender','Married','Education','Self_Employed'], aggfunc=lambda x: x.mode().iat[0])
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Dependents'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed']])
  data.loc[i,'Dependents'] = impute_grps.loc[ind].values[0]  
print(data.isnull().sum())
data = data[data.Dependents.notnull()]
impute_grps = data.pivot_table(values=['Credit_History'], index =['Gender','Married','Education','Self_Employed','Dependents'],  aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Credit_History'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed'],row['Dependents']])
  data.loc[i,'Credit_History'] = impute_grps.loc[ind].values[0]


data = data.fillna(0)

def percConvert(ser):
  return ser/float(ser[-1])

data.Gender = pd.Categorical(data.Gender)
data.Married = pd.Categorical(data.Married)
data.Dependents = pd.Categorical(data.Dependents)
data.Education = pd.Categorical(data.Education)
data.Self_Employed = pd.Categorical(data.Self_Employed)
data.Property_Area = pd.Categorical(data.Property_Area)
#data.Loan_Status = pd.Categorical(data.Loan_Status)

data_new = data.Loan_ID

data.drop('Loan_ID', axis=1, inplace=True)
print(data.dtypes)
for y in data.columns:
	if(data[y].dtypes.name == 'category'):
		print(data[y].name)
		print (data[y].unique())

df_Gender = pd.get_dummies(data['Gender'],prefix='Gender',drop_first=True)
df_Married = pd.get_dummies(data['Married'],prefix='Married',drop_first=True)
df_Dependents = pd.get_dummies(data['Dependents'],prefix='Dependents',drop_first=True)
df_Education = pd.get_dummies(data['Education'],prefix='Education',drop_first=True)
df_Self_Employed = pd.get_dummies(data['Self_Employed'],prefix='Self_Employed',drop_first=True)
print(df_Self_Employed.columns.names)
df_Property_Area = pd.get_dummies(data['Property_Area'],prefix='Property_Area',drop_first=True)
#df_Loan_Status = pd.get_dummies(data['Loan_Status'],prefix='Loan_Status',drop_first=True)

data = data.join(df_Gender)
data = data.join(df_Married)
data = data.join(df_Dependents)
data = data.join(df_Education)
data = data.join(df_Self_Employed)
df = data.join(df_Property_Area)
#df = data.join(df_Loan_Status)

df.drop('Gender', axis=1, inplace=True)
df.drop('Married', axis=1, inplace=True)
df.drop('Dependents', axis=1, inplace=True)
df.drop('Education', axis=1, inplace=True)
df.drop('Self_Employed', axis=1, inplace=True)
df.drop('Property_Area', axis=1, inplace=True)
#df.drop('Loan_Status', axis=1, inplace=True)
print(df.isnull().sum())
predictions = estimator.predict(df)
print(predictions)

import pandas as pd
df = pd.DataFrame(data_new)
result = pd.DataFrame(predictions)
df = df.join(result)
df.columns = ['Loan_ID', 'Loan_Status']
df.to_csv('result.csv', index=False, sep=',')

'''

pca = decomposition.PCA()
n_components = [2, 10, 14]	
pipeline1 = Pipeline(([('pca', pca),('clf', RandomForestClassifier())]))
pipeline2 = Pipeline(([('pca', pca),('clf', KNeighborsClassifier())]))
pipeline3 = Pipeline(([('pca', pca),('clf', SVC())]))
pipeline4 = Pipeline(([('pca', pca),('clf', MultinomialNB())]))
pipeline5 = Pipeline(([('pca', pca),('clf', ExtraTreesClassifier())]))
pipeline6 = Pipeline(([('pca', pca),('clf', RandomForestClassifier())]))
pipeline7 = Pipeline(([('pca', pca),('clf',  GaussianNB())]))
pipeline8 = Pipeline(([('pca', pca),('clf',  AdaBoostClassifier())]))
pipeline9 = Pipeline(([('pca', pca),('clf',  GradientBoostingClassifier())]))	
parameters1 = {
    'clf__n_estimators': [10, 20, 30],
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_features': [5, 10, 15],
    'clf__max_depth': ['auto', 'log2', 'sqrt', None]
}
parameters2 = {
    'clf__n_neighbors': [3, 7, 10],
    'clf__weights': ['uniform', 'distance']
}
parameters3 = {
    'clf__C': [0.01, 0.1, 1.0],
    'clf__kernel': ['rbf', 'poly'],
    'clf__gamma': [0.01, 0.1, 1.0],
}
parameters4 = {
    'clf__alpha': [0.01, 0.1, 1.0]
}
parameters5 = {
    'clf__n_estimators': [16, 32]
}	
parameters6 = {
    'clf__n_estimators': [16, 32]
}	
parameters7 = {
    'clf__priors': [None]
}	
parameters8 = {
    'clf__n_estimators': [16, 32]
}	
parameters9 = {
    'clf__n_estimators': [16, 32],
	'clf__learning_rate': [0.8, 1.0],
}	
params = { 
    'RandomForestClassifier': { 'n_estimators': [16, 32] }
}
params = params[0]
#pars = [parameters1, parameters2, parameters3, parameters4, parameters5, parameters6, parameters7, parameters8, parameters9]
#pips = [pipeline1, pipeline2, pipeline3, pipeline4, pipeline5, pipeline6, pipeline7, pipeline8, pipeline9]
pars = parameters5
pips = pipeline5
#for i in range(len(1)):
gs = GridSearchCV(pips, dict(pca__n_components=n_components, param_grid=params), cv=3, verbose=2, refit=False)
gs = gs.fit(X_train, y_train)
#print(gs.best_score_)
predictions = estimator.predict(X_test)
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print(classification_report(y_test, predictions))