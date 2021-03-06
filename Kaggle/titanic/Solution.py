import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
pd.options.display.max_columns = 1000
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
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
from sklearn.cross_validation import train_test_split

data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
print(data.head())
print(data.isnull().sum())

y = data.Survived
final_y = pd.DataFrame(data.Survived).copy()
data.drop('Survived', axis=1, inplace=True)
print(data.shape)
result = pd.concat([data, data_test], axis=0)
print(result.shape)
print(result.dtypes)

#result.Survived = pd.Categorical(result.Survived)
result.Pclass = pd.Categorical(result.Pclass)
result.Sex = pd.Categorical(result.Sex)
result.Embarked = pd.Categorical(result.Embarked)

result['FamilySize'] = result['Parch'] + result['SibSp'] + 1
result['Singleton'] = result['FamilySize'].map(lambda s : 1 if s == 1 else 0)
result['SmallFamily'] = result['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
result['LargeFamily'] = result['FamilySize'].map(lambda s : 1 if 5<=s else 0)

result['Embarked'] = result['Embarked'].fillna(result['Embarked'].mode()[0])


def get_titles():

    global combined
    
    # we extract the title from each name
    result['Title'] = result['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    result['Title'] = result.Title.map(Title_Dictionary)
get_titles()

def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    result.Age = result.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    #status('age')
process_age()

result["Fare"].fillna(result["Fare"].median(), inplace=True)

result.drop('Parch', axis=1, inplace=True)
result.drop('SibSp', axis=1, inplace=True)
result.drop('FamilySize', axis=1, inplace=True)
result.drop('Name', axis=1, inplace=True)
result.drop('Cabin', axis=1, inplace=True)
result.drop('Ticket', axis=1, inplace=True)
result.drop('PassengerId', axis=1, inplace=True)

'''
#print orrelation matrix
print(result.corr())


result1 = result.copy()
train = result1.iloc[:891,]
test = result1.iloc[892:,:]
dataplot = pd.concat([train, pd.DataFrame(y.copy())], axis=1)

for y in dataplot.columns:
	if(dataplot[y].dtypes.name == 'category'):
		print(dataplot[y].name)
		print (dataplot[y].unique())

print(dataplot.isnull().sum())'''
'''
#People of class 1 has high fare
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(8,5), ax = ax)

#people of class 1 are less dead
survived_sex = data[data['Survived']==1]['Pclass'].value_counts()
dead_sex = data[data['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(8,5))

#Males are more dead and females more survived
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(8,5))

#People in age of 16-32 are more dead, followed by 17-54, then 0-15 and last 54 above
figure = plt.figure(figsize=(8,5))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.legend()
plt.show()
'''

data = result
# encoding into 3 categories:
pclass_dummies = pd.get_dummies(data['Pclass'],prefix="Pclass",drop_first=True)
# adding dummy variables
data = pd.concat([data,pclass_dummies],axis=1)    
data.drop('Pclass',axis=1,inplace=True)

sex_dummies = pd.get_dummies(data['Sex'],prefix="Sex", drop_first=True)
# adding dummy variables
data = pd.concat([data,sex_dummies],axis=1)    
data.drop('Sex',axis=1,inplace=True)
print(data.dtypes)

# convert from float to int
data['Age'] = data['Age'].astype(int)

data.loc[ data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[(data['Age'] > 64), 'Age'] = 4

data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[ data['Fare'] > 31, 'Fare'] = 3

# convert from float to int
data['Fare'] = data['Fare'].astype(int)
def title_map(title):
    if title in ['Mr']:
        return 1
    elif title in ['Master']:
        return 3
    elif title in ['Ms','Mlle','Miss']:
        return 4
    elif title in ['Mme','Mrs']:
        return 5
    else:
        return 2
data['Title'] = data['Title'].apply(title_map)
titles_dummies = pd.get_dummies(data['Title'],prefix='Title',drop_first=True)
data = pd.concat([data,titles_dummies],axis=1)    
# removing the title variable
data.drop('Title',axis=1,inplace=True)

def embarked_map(title):
    if title in ['S']:
        return 1
    elif title in ['C']:
        return 2
    elif title in ['Q']:
        return 3 
data['Embarked'] = data['Embarked'].apply(embarked_map)
embarked_dummies = pd.get_dummies(data['Embarked'],prefix='Embarked',drop_first=True)
data = pd.concat([data,embarked_dummies],axis=1)
data.drop('Embarked',axis=1,inplace=True)

age_dummies = pd.get_dummies(data['Age'],prefix="Age", drop_first=True)
# adding dummy variables
data = pd.concat([data,age_dummies],axis=1)    
data.drop('Age',axis=1,inplace=True)
print(data.dtypes)

#scaler = MinMaxScaler() 
#scaled_values = scaler.fit_transform(data) 
#data.loc[:,:] = scaled_values

result1 = data.copy()
train = data.iloc[:891,]
test = data.iloc[891:,:]
print(test.shape)
#print(final_y)
#finaldata = pd.concat([train, pd.DataFrame(final_y)], axis=1)
		
#print(finaldata.head())

X_train = train
print(X_train.shape)
Y_train = final_y
print(final_y.shape)
X_test  = test

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [50,100,120,150],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(np.reshape(Y_train.values,[891,]) , n_folds=5)
random_forest = GridSearchCV(random_forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

random_forest.fit(X_train, Y_train)

Y_pred_1 = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))
Y_pred = Y_pred_1
print(Y_pred)

print(data_test["PassengerId"].shape)
print(Y_pred.shape)
submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"].values,
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)