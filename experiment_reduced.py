# # 1. IMPORT MODULE 
# **1.1. Import pandas** 
import pandas as pd
import numpy as np


# **Import Machine Learning Module** 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report, confusion_matrix


# ** import packages for imbalance-learn for balancing class**
from imblearn.over_sampling import SMOTE

# ** import utilites packages
import pickle
import cdsw


# # 2. IMPORT DATASET 

# **Import dataset**
sample_train = pd.read_csv('sample_data/sample_train.csv')
sample_bureau = pd.read_csv('sample_data/sample_bureau.csv')
sample_prev_app = pd.read_csv('sample_data/sample_prev_app.csv')


# # 3. DATA AGGREGATION OF SAMPLE_BUREAU

# **Count the number of previous loans**
previous_loan_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})


# **Count Credit Active**
credit_active_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['CREDIT_ACTIVE'].count().rename(columns = {'CREDIT_ACTIVE':'credit_active_counts'})



# # 4. DATA AGGREGATION OF SAMPLE_PREVIOUS_APPLICATION

# **Numeric Aggregating**
prev_agg = sample_prev_app.drop(['SK_ID_PREV'], axis = 1).groupby('SK_ID_CURR', as_index = False)['AMT_CREDIT','AMT_APPLICATION'].agg(['mean', 'max', 'min', 'sum']).reset_index()
prev_agg.columns = [''.join(col) for col in prev_agg.columns]


# # 5. DATA MERGING 

# **Merge previous_loan_counts with sample_train**
sample_train_new1 = sample_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')


# **Merge prev_aggregating with sample_train**
sample_train_new2 = sample_train_new1.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')

  

#Null_Value(sample_train_new2)  
  
# # 6. MANUAL FEATURE SELECTION

# **Make function to choose suitable feature **
# define feature location that wanna removed
thr_train = 0.6
size_train=sample_train_new2.shape[0]
dropcol_train = []

#looping to take the number of null of every feature 
for col in sample_train_new2.columns :
    if (sample_train_new2[col].isnull().sum()/size_train >= thr_train):
        dropcol_train.append(col)
        

# Drop feature greater than > 60 % that contain null value
sample_train_new3 = sample_train_new2.drop(dropcol_train, axis = 1)


# Recheck dataframe that has been choosed
#Null_Value(sample_train_new3)

# # 7. FILLING MISSING VALUE

# to check for each feature is object or numeric train
categorical_list = []
numerical_list = []
for col in sample_train_new3.columns.tolist():
    if sample_train_new3[col].dtype=='object':
        categorical_list.append(col)
    else:
        numerical_list.append(col)


# 1. Filling Missing Values in Categorical dataframe
for col in categorical_list:
  sample_train_new3[col] = sample_train_new3[col].fillna(sample_train_new3[col].mode().iloc[0])

# 2. Filling Missing Values in Numerical dataframe
for col in numerical_list:
  sample_train_new3[col] = sample_train_new3[col].fillna(sample_train_new3[col].median())
  

sample_train_new3.info()

sample_train_new4 = sample_train_new3[['FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','TARGET']]

sample_train_new4.head()
sample_train_new4.info()

# # 8. ONE HOT-ENCODING
# Get dummy features
df = pd.get_dummies(sample_train_new4, drop_first=True)

df.head()

# # 9. BALANCING CLASS

# Split data to train and test
X = df.drop(['TARGET'], axis=1)
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Balancing Class Target with SMOTE Method
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# # Register CDSW Parameter Tracking
# Set Parameter

param_numTrees = 10 #int(sys.argv[1]) #10
param_maxDepth = 5 #int(sys.argv[2]) #5
param_impurity = 'gini' #sys.argv[3] #'gini'

#cdsw.track_metric("numTrees",param_numTrees)
#cdsw.track_metric("maxDepth",param_maxDepth)
#cdsw.track_metric("impurity",param_impurity)


# # 10. DEVELOP MODEL

# Develop Machine learning model Using RandomForest Classifier 
rf = RandomForestClassifier(n_jobs=10,
                             n_estimators=param_numTrees, 
                             max_depth=param_maxDepth, 
                             criterion = param_impurity,
                             random_state=0) 
rf.fit(X_train_res,y_train_res) # training model 


# # 11. MODEL EVALUATION
# Predict Model
pred_rf_test = rf.predict(X_test)
    
#cdsw.track_metric("accuracy", accuracy_score(y_test, pred_rf_test))
print(accuracy_score(y_test, pred_rf_test))

probs = rf.predict_proba(X_test)
probs = probs[:, 1]

#cdsw.track_metric("auc", roc_auc_score(y_test, probs))
print(roc_auc_score(y_test, probs))

pickle.dump(rf, open("sklearn_rf.pkl","wb"))

cdsw.track_file("sklearn_rf.pkl")
