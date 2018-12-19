# # 1. IMPORT MODULE test masukin comment
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


# **1.2. import packages for imbalance-learn for balancing class**
from imblearn.over_sampling import SMOTE


# **1.3. visualization**
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

# # 2. IMPORT DATASET 

# **Import dataset**
sample_train = pd.read_csv('sample_data/sample_train.csv')
sample_bureau = pd.read_csv('sample_data/sample_bureau.csv')
sample_prev_app = pd.read_csv('sample_data/sample_prev_app.csv')




# **Preview of Dataset**
sample_train.head()
sample_bureau.head()
sample_prev_app.head()

sample_train.columns

# # 3. EDA (EXploratory Data Anaysis)

# **Descriptive of sample_train**
sample_train.describe()

# **Distribution of AMT_CREDIT**
plt.figure(figsize=(8,5))
sns.distplot(sample_train["AMT_CREDIT"]).set_title('Distribution of AMT_CREDIT')

# **Distibution of AMT_INCOME_TOTAL**
plt.figure(figsize=(8,5))
sns.distplot(sample_train["AMT_INCOME_TOTAL"]).set_title('Distribution of AMT_INCOME_TOTAL')

# **Distribusi of AMT_GOODS_PRICE**
plt.figure(figsize=(8,5))
sns.distplot(sample_train["AMT_GOODS_PRICE"].dropna()).set_title('Distribution of AMT_GOODS_PRICE')

# **Data is balanced or imbalanced**
temp = sample_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
plt.figure(figsize = (6,6))
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df).set_title('Application loans repayed - train dataset')

# **Family Status of Applicant's who applied for loan**
temp = sample_train["NAME_FAMILY_STATUS"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
plt.figure(figsize = (6,6))
sns.barplot(x = 'labels', y="values", data=df).set_title('Family Status')

# **Income sources of Applicant's who applied for loan**
temp = sample_train["NAME_INCOME_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
sns.barplot(x = 'labels', y="values", data=df).set_title('Income sources of Applicant')

# # 4. CHECKING MISSING DATA

# **make NUll function for checking each Dataset**
def Null_Value(df):
  total = df.isnull().sum().sort_values(ascending = False)
  percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
  missing  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  return missing.head(10)

# **checking Missing Value**
Null_Value(sample_train)
Null_Value(sample_bureau)
Null_Value(sample_prev_app)

# # 5. DATA AGGREGATION OF SAMPLE_BUREAU

# **Count the number of previous loans**
previous_loan_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()

# **Count Credit Active**
credit_active_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['CREDIT_ACTIVE'].count().rename(columns = {'CREDIT_ACTIVE':'credit_active_counts'})
credit_active_counts.head()


# # 6. DATA AGGREGATION OF SAMPLE_PREVIOUS_APPLICATION

# **Numeric Aggregating**
prev_agg = sample_prev_app.drop(['SK_ID_PREV'], axis = 1).groupby('SK_ID_CURR', as_index = False)['AMT_CREDIT','AMT_APPLICATION'].agg(['mean', 'max', 'min', 'sum']).reset_index()
prev_agg.columns = [''.join(col) for col in prev_agg.columns]
prev_agg.head()


# # 7. DATA MERGING 

# **Merge previous_loan_counts with sample_train**
sample_train_new1 = sample_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
sample_train_new1.head()

# **Merge prev_aggregating with sample_train**
sample_train_new2 = sample_train_new1.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')
sample_train_new2.head()
  

Null_Value(sample_train_new2)  
  
# # 8. MANUAL FEATURE SELECTION

# **Make function to choose suitable feature **
# define feature location that wanna removed
thr_train = 0.6
size_train=sample_train_new2.shape[0]
dropcol_train = []

#looping to take the number of null of every feature 
for col in sample_train_new2.columns :
    if (sample_train_new2[col].isnull().sum()/size_train >= thr_train):
        dropcol_train.append(col)
        
print('Feature dataframe will be removed = ', '\n', dropcol_train)

# Drop feature greater than > 60 % including null value
sample_train_new3 = sample_train_new2.drop(dropcol_train, axis = 1)
sample_train_new3.head()

# Recheck dataframe that has been choosed
Null_Value(sample_train_new3)

# # 9. FILLING MISSING VALUE

# to check for each feature is object or numeric train
categorical_list = []
numerical_list = []
for col in sample_train_new3.columns.tolist():
    if sample_train_new3[col].dtype=='object':
        categorical_list.append(col)
    else:
        numerical_list.append(col)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

# Recheck again
Null_Value(sample_train_new3)

# 1. Filling Missing Values in Categorical dataframe
for col in categorical_list:
  sample_train_new3[col] = sample_train_new3[col].fillna(sample_train_new3[col].mode().iloc[0])

# 2. Filling Missing Values in Numerical dataframe
for col in numerical_list:
  sample_train_new3[col] = sample_train_new3[col].fillna(sample_train_new3[col].median())
  

# # 10. ONE HOT-ENCODING
# Get dummy features
df = pd.get_dummies(sample_train_new3, drop_first=True)

# # 11. BALANCING CLASS

# Split data to train and test
X = df.drop(['TARGET'], axis=1)
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Balancing Class Target with SMOTE Method
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# # 11. DEVELOP MODEL

# Develop Machine learning model Using RandomForest Classifier 
rf = RandomForestClassifier() # default parameter
rf.fit(X_train_res,y_train_res) # training model 

# # 13. MODEL EVALUATION
# Predict Model
pred_rf_train = rf.predict(X_train_res)
pred_rf_test = rf.predict(X_test)


#Make Function Model Evaluation
def eval(target, predict):
    print('Confusion Matrix : ')
    print(confusion_matrix(target, predict),'\n')
    print('Classification Report : ')
    print(classification_report(target, predict))
    print('Accuracy Model : ',accuracy_score(target, predict)*100,'%')


# Train Evaluation
eval(y_train_res,pred_rf_train)

# Test Evaluation
eval(y_test,pred_rf_test)
    
# ROC curve
# Make Function ROC CURVE 
def ROC_CURVE(feature, target, model):
    probs = model.predict_proba(feature)
    probs = probs[:, 1]
    auc = roc_auc_score(target, probs)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(target, probs)
    #sixe of figure 
    plt.figure(figsize=(10,6))
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')

ROC_CURVE(X_test, y_test, rf)

# # 14. TUNING HYPERPARAMETER USING RANDOMIZE SEARCH

# Setting Parameter 
n_estimators = [100, 500, 1000]
max_features = [10, 20, 50]
max_depth = [5, 10,20]
min_samples_split = [2, 5, 10]
min_samples_leaf = [10, 20, 30]
bootstrap = [True, False]

# Create the random grid
param = {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'bootstrap': bootstrap}

# Training model
clf = RandomizedSearchCV(rf, param, n_iter=3, cv= 5)

# fit randomize search 
best_model = clf.fit(X_train_res,y_train_res)

#predict best model
pred_best_train = best_model.predict(X_train_res) 
pred_best_test = best_model.predict(X_test)

# Train Evaluation
eval(y_train_res,pred_best_train)

# Test Evaluation
eval(y_test,pred_best_test)

# ROC Curve
ROC_CURVE(X_test, y_test, clf)