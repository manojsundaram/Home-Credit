# # Import Pandas DF operation

import pandas as pd # Import modul pandas  
import numpy as np  # Import modul numpy 
# ______________________________________________________________________




# **Import Machine Learning Module** 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report, confusion_matrix

# => Import modul machine learning yaitu Scikit-Learn(Sklearn)
# ______________________________________________________________________




# # Balancing Class 
from imblearn.over_sampling import SMOTE

# => Import modul balancing class/target menggunakan metode metode SMOTE 
# ______________________________________________________________________


#  import utilites packages
import pickle
import cdsw




# IMPORT DATASET
sample_train = pd.read_csv('sample_data/sample_train.csv')
sample_bureau = pd.read_csv('sample_data/sample_bureau.csv')
sample_prev_app = pd.read_csv('sample_data/sample_prev_app.csv')

# => import data dalam bentuk csv   
# ________________________________________________________________________________




# PREVIEW DATASET 
sample_train.head()
sample_bureau.head()
sample_prev_app.head()

# => code diatas digunakan untuk melihat display pandas dataframe
# ________________________________________________________________________________


#change XNA subset in CODE_GENDER feature 
sample_train.loc[sample_train['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = 'M'

# => Mengubah subset yang ada di feature 'CODE_GENDER' = 'XNA' menjadi M
# ________________________________________________________________________________



# # DATA AGGREGATION OF SAMPLE_BUREAU

# Count the number of previous loans
previous_loan_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
# Count Credit Active
credit_active_counts = sample_bureau.groupby('SK_ID_CURR', as_index=False)['CREDIT_ACTIVE'].count().rename(columns = {'CREDIT_ACTIVE':'credit_active_counts'})

# => code diatas digunakan untuk melakukan agregasi berdasarkan feature tertentu 
# ________________________________________________________________________________





# # DATA AGGREGATION OF SAMPLE_PREVIOUS_APPLICATION

# Numeric Aggregating
prev_agg = sample_prev_app.drop(['SK_ID_PREV'], axis = 1).groupby('SK_ID_CURR', as_index = False)['AMT_CREDIT','AMT_APPLICATION'].agg(['mean', 'max', 'min', 'sum']).reset_index()
prev_agg.columns = [''.join(col) for col in prev_agg.columns]

# => Dalam hal ini kita dapat melakukan agregasi dengan melihat mean dari setiap feature berdasarkan feature tertentu
# ___________________________________________________________________________________________________________________



# # DATA MERGING 

# Merge previous_loan_counts with sample_train
sample_train_new1 = sample_train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
# Merge prev_aggregating with sample_train
sample_train_new2 = sample_train_new1.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')

# => Code diatas digunakan untuk menggabungkan column dari tabel lain ke dalam tabel yang diinginkan
# __________________________________________________________________________________________________

  
  
# # MANUAL FEATURE SELECTION

# Choose feature with null => 60 %
thr_train = 0.6
size_train=sample_train_new2.shape[0]
dropcol_train = []

#looping to take the number of null of every feature 
for col in sample_train_new2.columns :
    if (sample_train_new2[col].isnull().sum()/size_train >= thr_train):
        dropcol_train.append(col)
        
# => Code diatas merupakan fungsi otomatis untuk memilih feature yang memiliki null value => 60 %
# __________________________________________________________________________________________________

        
  
        
# Drop feature greater than > 60 % that contain null value
df = sample_train_new2.drop(dropcol_train, axis = 1)
# => Drop feature yang memiliki null value => 60 % karena informasi dalam feature tersebut terlalu berpengaruh 
# _____________________________________________________________________________________________________________



# # FILLING MISSING VALUE

# to check for each feature is object or numeric train
categorical_list = []
numerical_list = []
for col in df.columns.tolist():
    if df[col].dtype=='object':
        categorical_list.append(col)
    else:
        numerical_list.append(col)

# 1. Filling Missing Values in Categorical dataframe
for col in categorical_list:
  df[col] = df[col].fillna(df[col].mode().iloc[0])
  
# => Mengisi categorical data null value dengan menggunakan modus  
# _____________________________________________________________________________________________________________


# 2. Filling Missing Values in Numerical dataframe
for col in numerical_list:
  df[col] = df[col].fillna(df[col].median())

# => Mengisi numerical data null value dengan menggunakan median  
# _____________________________________________________________________________________________________________


  

# Choosing some of feature for train model
df_train = df[['NAME_INCOME_TYPE',
               'NAME_FAMILY_STATUS',
               'NAME_EDUCATION_TYPE',
               'REG_CITY_NOT_WORK_CITY',
               'FLAG_PHONE',
               'FLAG_OWN_CAR',
               'CODE_GENDER',
               'FLAG_OWN_REALTY',
               'EXT_SOURCE_3',
               'EXT_SOURCE_2',
               'AMT_INCOME_TOTAL',
               'AMT_CREDIT',
               'TARGET']]

# => Pilih feature yang diperlukan untuk develop model
# _____________________________________________________________________________________________________________


# # ONE HOT-ENCODING
# Get dummy features
df_train = pd.get_dummies(df_train, drop_first=True)

# => Code diatas digunakan untuk membuat kode untuk setiap subset yang tipe datanya kategorik  
# __________________________________________________________________________________________________



# # BALANCING CLASS

# Split data to train and test
X = df_train.drop(['TARGET'], axis=1)
y = df_train['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# => Lakukan split data kedalam data train dan test dengan perbandngan (80:20)
# => pembagian tersebut dengan memisahkan antara semua feature dengan targetnya
# __________________________________________________________________________________________________




# Balancing Class Target with SMOTE Method
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# => Karena target nya tidak balance maka dilakukan balancing terlebih dahulu untuk data train
# __________________________________________________________________________________________________




# # Register CDSW Parameter Tracking
# Set Parameter

param_numTrees = 10 #int(sys.argv[1]) #10
param_maxDepth = 5 #int(sys.argv[2]) #5
param_impurity = 'gini' #sys.argv[3] #'gini'

#cdsw.track_metric("numTrees",param_numTrees)
#cdsw.track_metric("maxDepth",param_maxDepth)
#cdsw.track_metric("impurity",param_impurity)

# => Penentuan parameter dalam algoritma Randomforest
# __________________________________________________________________________________________________





# # DEVELOP MODEL

# Develop Machine learning model Using RandomForest Classifier 
rf = RandomForestClassifier(n_jobs=10,
                             n_estimators=param_numTrees, 
                             max_depth=param_maxDepth, 
                             criterion = param_impurity,
                             random_state=0) 
rf.fit(X_train_res,y_train_res) # training model 

# => Model yang digunakan adalah algoritma Randomforest
# __________________________________________________________________________________________________




# # 11. MODEL EVALUATION
# Predict Model
pred_rf_test = rf.predict(X_test)

# => melaukan prediksi dengan data test 
# __________________________________________________________________________________________________



#cdsw.track_metric("accuracy", accuracy_score(y_test, pred_rf_test))
print(accuracy_score(y_test, pred_rf_test))

# => Evaluasi model yang dibuat dengan melihat tingkat akurasi yang dihasilkan 
# __________________________________________________________________________________________________


probs = rf.predict_proba(X_test)
probs = probs[:, 1]

#cdsw.track_metric("auc", roc_auc_score(y_test, probs))
print(roc_auc_score(y_test, probs))

# => Evaluasi model yang dibuat dengan melihat nilai Area Under Curve (AUC) 
# __________________________________________________________________________________________________



pickle.dump(rf, open("sklearn_rf_large.pkl","wb"))

cdsw.track_file("sklearn_rf_large.pkl")

# => Save model
# __________________________________________________________________________________________________

