import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import sys

###################################################Reading the dataset###############################################################
train_df = pd.read_csv('train.csv', sep=',')
test_df = pd.read_csv('test.csv', sep=',')
test = test_df.copy()
# label
train_y = np.array(train_df["Purchase"])
# Saving id variables to create final submission
ids_test = test['User_ID'].copy()
product_ids_test = test['Product_ID'].copy()

print(train_df.head())
print(train_df.isnull().sum())


###################################Univariate and Bivariate Extrapolation Analysis##################################################
figure = plt.figure(figsize=(8,5))
sns.set(style="whitegrid", color_codes=True)

'''
sns.stripplot(x="Product_Category_1", y="Purchase",hue="Gender", jitter=True, data=train_df);
sns.stripplot(x="Product_Category_1", y="Purchase",hue="Age",  data=train_df, jitter=0.5,dodge=True);
sns.stripplot(x="Product_Category_1", y="Purchase",hue="Gender",  data=train_df, jitter=0.5,dodge=True);
plt.xlabel('Product_Category_1')
plt.ylabel('Purchase ($)')
plt.legend()



train_df.plot(kind="scatter",     # Create a scatterplot
              x="Product_Category_1",          # Put carat on the x axis
              y="Purchase",          # Put price on the y axis
              figsize=(8,5),
              ylim=(0,25000)) 
			  
sns.boxplot(x="Product_Category_1", y="Purchase", hue="Gender", data=train_df);

sns.pairplot(vars=["Gender"], data=train_df, hue="Product_Category_1", size=5)

sns.barplot(x="Product_Category_1", y="Purchase", hue="Age", data=train_df);
sns.pairplot(train_df,x_vars=["Purchase", "Age"], y_vars=["Product_Category_1", "Product_Category_1"])
plt.show()		
'''
###############################Features Cleaning, Normalisation, Selection and Extraction###############################################
#label encoder - One hot Encoding
le = LabelEncoder()

diff_colums = list(set(pd.unique(test_df['Product_ID'])) - set(pd.unique(train_df['Product_ID'])))
print(diff_colums)

train_df['Product_Category_2'].fillna(0, inplace=True)
test_df['Product_Category_2'].fillna(0, inplace=True)

train_df['Product_Category_3'].fillna(0, inplace=True)
test_df['Product_Category_3'].fillna(0, inplace=True)

train_df['Product_Category_2'] = train_df['Product_Category_2'].astype(np.int)
test_df['Product_Category_2'] = test_df['Product_Category_2'].astype(np.int)

train_df['Product_Category_3'] = train_df['Product_Category_3'].astype(np.int)
test_df['Product_Category_3'] = test_df['Product_Category_3'].astype(np.int)


train_df['Gender'] = train_df['Gender'].map({'F':0,'M':1}).astype(np.int)
#train_df.Gender = pd.Categorical(train_df.Gender)
test_df['Gender'] = test_df['Gender'].map({'F':0,'M':1}).astype(np.int)
#test_df.Gender = pd.Categorical(test_df.Gender)

train_df.Age = pd.Categorical(train_df.Age)
train_df['Age'] = le.fit_transform(train_df['Age'])
test_df.Age = pd.Categorical(test_df.Age)
test_df['Age'] = le.fit_transform(test_df['Age'])

train_df.City_Category = pd.Categorical(train_df.City_Category)
train_df['City_Category'] = le.fit_transform(train_df['City_Category'])
test_df.City_Category = pd.Categorical(test_df.City_Category)
test_df['City_Category'] = le.fit_transform(test_df['City_Category'])

train_df.Stay_In_Current_City_Years = pd.Categorical(train_df.Stay_In_Current_City_Years)
train_df['Stay_In_Current_City_Years'] = le.fit_transform(train_df['Stay_In_Current_City_Years'])
test_df.Stay_In_Current_City_Years = pd.Categorical(test_df.Stay_In_Current_City_Years)
test_df['Stay_In_Current_City_Years'] = le.fit_transform(test_df['Stay_In_Current_City_Years'])

train_df.Product_ID = pd.Categorical(train_df.Product_ID)
test_df.Product_ID = pd.Categorical(test_df.Product_ID)

train_df['product_concat'] = train_df['Product_Category_1'].astype(str) +'_'+ train_df['Product_Category_2'].astype(str) +'_'+ train_df['Product_Category_3'].astype(str)
test_df['product_concat'] = test_df['Product_Category_1'].astype(str) +'_'+ test_df['Product_Category_2'].astype(str) +'_'+ test_df['Product_Category_3'].astype(str)

# Label Encoding User_IDs
train_df['product_concat'] = le.fit_transform(train_df['product_concat'])
test_df['product_concat'] = le.fit_transform(test_df['product_concat'])

product_id_res = train_df.groupby(["Product_ID"])["Purchase"].mean()
avg_cost = train_df["Purchase"].mean()
# If i find a product id for which i dont have an avg pricing i will use global vg pricing.
product_id_res_map = {}
# created a map with product id to avg price map
val = product_id_res.iteritems()
for key, value in val:
    p_id = str(key)
    product_id_res_map[p_id] = value


def get_purchase_mean(product_id, product_category=None, key=None):
    key_pair = str(product_id)
    key_pair_pid = str(product_id) + str(product_category)
    if key_pair in product_id_res:
        return product_id_res[key_pair]
    return avg_cost

# Create a feature with pruduct_id to avg price of that product map
train_df["purchase_avg_by_p_id"] = train_df.Product_ID.apply(get_purchase_mean).astype(np.int)
test_df["purchase_avg_by_p_id"] = test_df.Product_ID.apply(get_purchase_mean).astype(np.int)

user_id_to_category_map = {}
customer_purchase_power = train_df.groupby("User_ID")["Purchase"].sum().values

'''
sns.distplot(customer_purchase_power,bins=20, kde=False);
plt.show()	
'''

customer_purchase_power = train_df.groupby("User_ID")["Purchase"].sum()
values = customer_purchase_power.iteritems()

for key, val in values:
    if val <= 555000:
        user_id_to_category_map[key] = 1
    elif val>555000 & val <= 1085550:
        user_id_to_category_map[key] = 2
    elif val>1085550 & val <= 1606680:
        user_id_to_category_map[key] = 3
    elif val>1606680 & val <= 2127810:
        user_id_to_category_map[key] = 4
    elif val>2127810 & val <= 2650000:
        user_id_to_category_map[key] = 5
    elif val>2650000 & val <= 3188000:
        user_id_to_category_map[key] = 6
    elif val > 3188000.0:
        user_id_to_category_map[key] = 7


def get_customer_category(user_id):
    if user_id in user_id_to_category_map:
        return user_id_to_category_map[user_id]
    return 1

# categorical columns that i chose
categorical_columns = ["Product_ID"]

train_df = train_df.drop(['Product_Category_1'], axis=1)
train_df = train_df.drop(['Product_Category_2'], axis=1)
train_df = train_df.drop(['Product_Category_3'], axis=1)
test_df = test_df.drop(['Product_Category_1'], axis=1)
test_df = test_df.drop(['Product_Category_2'], axis=1)
test_df = test_df.drop(['Product_Category_3'], axis=1)

# Tagged each user with a category id
train_df['user_category'] = train_df.User_ID.apply(get_customer_category)
test_df['user_category'] = test_df.User_ID.apply(get_customer_category)


temp_y = train_df.Purchase
train_df.drop('Purchase', axis=1, inplace=True)
result = pd.concat([train_df, test_df], axis=0)
result.User_ID = pd.Categorical(result.User_ID)
result['User_ID'] = le.fit_transform(result['User_ID'])
result['Product_ID'] = le.fit_transform(result['Product_ID'])

train = result.iloc[:550068,:]
test = result.iloc[550068:,:]
print(train.head())
train['Purchase'] = pd.DataFrame(temp_y)
train_df = train
test_df = test

'''
# Encoding categorical variable with label encoding
for var in categorical_columns:
    lb = preprocessing.LabelEncoder()
    full_var_data = pd.concat((train_df[var], test_df[var]), axis=0).astype('str')
    lb.fit(full_var_data)
    train_df[var] = lb.transform(train_df[var].astype('str'))
    test_df[var] = lb.transform(test_df[var].astype('str'))
'''	
print(train_df.dtypes)
print(test_df.dtypes)
print(train_df.head())
print(test_df.head())

##########################################################Modeling and Prediction on train and test dataset############################################################
y = train_df['Purchase']
train_df.drop(['Purchase'], inplace=True, axis=1)


# Modeling
dtrain = xgb.DMatrix(train_df.values, label=y, missing=np.nan)

param = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
		 'max_depth': 10, 'eta': 0.1, 'nthread': 4,
		 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
		 'max_delta_step': 0, 'gamma': 0}
num_round = 690

#seeds = [1122, 2244, 3366, 4488, 5500]
seeds = [1122]
test_preds = np.zeros((len(test_df), len(seeds)))

# Prediction
for run in range(len(seeds)):
	sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seeds)))
	sys.stdout.flush()
	param['seed'] = seeds[run]
	clf = xgb.train(param, dtrain, num_round)
	dtest = xgb.DMatrix(test_df.values, missing=np.nan)
	test_preds[:, run] = clf.predict(dtest)

#Saving the model 
test_preds = np.mean(test_preds, axis=1)
submit = pd.read_csv("Sample_Submission_Tm9Lura.csv")
# Submission file
submit = pd.DataFrame({'User_ID': ids_test, 'Product_ID': product_ids_test, 'Purchase': test_preds})
submit = submit[['User_ID', 'Product_ID', 'Purchase']]
submit['Purchase'] = submit['Purchase'].astype(np.int)
print(submit.isnull().sum())
submit.ix[submit['Purchase'] < 0, 'Purchase'] = 12  # changing min prediction to min value in train
submit.to_csv("final.csv", index=False)