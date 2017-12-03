import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from numpy import nan
from sklearn.cross_validation import train_test_split

data = pd.read_csv('train.csv', sep=',')
data.replace('nan', nan, inplace=True)

#print(data.head())
print(data.isnull().sum())
#print( data.dtypes)

#print(mode(data['Gender']).mode[0])

data['Gender'].fillna('Male', inplace=True)
data['Married'].fillna('Yes', inplace=True)
data['Self_Employed'].fillna('No', inplace=True)
print(data.isnull().sum())
impute_grps = data.pivot_table(values=['LoanAmount'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
print(impute_grps)
print(data.loc[data['LoanAmount'].isnull(),:])
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]

#Now check the #missing values again to confirm:
print(data.isnull().sum())


impute_grps = data.pivot_table(values=['Loan_Amount_Term'], index =['Gender','Married','Self_Employed'], aggfunc=np.mean)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Loan_Amount_Term'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'Loan_Amount_Term'] = impute_grps.loc[ind].values[0]

impute_grps = data.pivot_table(values=['Dependents'], index =['Gender','Married','Education','Self_Employed'], aggfunc=lambda x: x.mode().iat[0])
print(impute_grps)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Dependents'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed']])
  data.loc[i,'Dependents'] = impute_grps.loc[ind].values[0]
  
impute_grps = data.pivot_table(values=['Credit_History'], index =['Gender','Married','Education','Self_Employed','Dependents'], aggfunc=lambda x: x.mode().iat[0])
print(impute_grps)
#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['Credit_History'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Education'],row['Self_Employed'],row['Dependents']])
  data.loc[i,'Credit_History'] = impute_grps.loc[ind].values[0]
  
#Now check the #missing values again to confirm:
print(data.isnull().sum())

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

bp = data.boxplot(column='LoanAmount', by=['Gender','Married','Education'])
plt.show()

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

print(df.dtypes)
#le_sex = preprocessing.LabelEncoder()
#to convert into numbers
#data.Gender = le_sex.fit_transform(data.Gender)
#print(data.head())


import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()

y_digits = df.Loan_Status
df.drop('Loan_Status', axis=1, inplace=True)
X_digits = df
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=0, test_size=0.25)

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
#confusion_matrix = confusion_matrix(y_test, predictions)
#print(confusion_matrix)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(estimator.score(X_test, y_test)))

#from sklearn.metrics import classification_report
#print(classification_report(y_test, predictions))