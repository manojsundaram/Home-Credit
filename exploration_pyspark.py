# # Import pandas 
import pandas as pd
import numpy as np

# # Import Pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import first, collect_list, mean, countDistinct
from pyspark.sql import functions as f
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql.functions import isnull, when, count, col
from pyspark.sql.types import IntegerType, StringType, DoubleType, ShortType

# # Import Machine Learning Module 
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report, confusion_matrix

# # import packages for imbalance-learn for balancing class
from imblearn.over_sampling import SMOTE


# visualization
import matplotlib.pyplot as plt 
import seaborn as sns 


# Create Spark Session
spark = SparkSession.builder.appName("Risk Model").getOrCreate()


# Import dataset
all_bureau = spark.table("poc_pengadaian.bureau")
all_train = spark.table("poc_pengadaian.application_train")
all_prev_app = spark.table("poc_pengadaian.previous_application2")

# Preview of Dataset
all_bureau.show(2)
all_train.show(2)
all_prev_app.show(2)

# Schema of Dataframe
all_bureau.printSchema()
all_train.printSchema()
all_prev_app.printSchema()

#Count row and column number 
print((all_bureau.count(), len(all_bureau.columns)))
print((all_train.count(), len(all_train.columns)))
print((all_prev_app.count(), len(all_prev_app.columns)))


# Select column from all_bureau 
df_bureau = all_bureau.select('sk_id_curr','credit_active','sk_id_bureau','amt_credit_sum')


# Change dtypes column all_bureau
df_bureau = df_bureau.withColumn("sk_id_curr", df_bureau["sk_id_curr"].cast(IntegerType()))
df_bureau = df_bureau.withColumn("sk_id_curr", df_bureau["credit_active"].cast(StringType()))
df_bureau = df_bureau.withColumn("sk_id_curr", df_bureau["sk_id_bureau"].cast(IntegerType()))
df_bureau = df_bureau.withColumn("amt_credit_sum", df_bureau["amt_credit_sum"].cast(DoubleType()))


# Select column from all_train 
df_train = all_train.select('sk_id_curr','target','ext_source_3','amt_credit','amt_income_total','ext_source_2','name_education_type','code_gender','flag_own_car','reg_city_not_work_city','flag_phone','name_family_status','name_income_type','commonarea_medi','commonarea_avg','occupation_type')

# Change dtypes column all_train
df_train = df_train.withColumn("sk_id_curr",df_train["sk_id_curr"].cast(IntegerType()))
df_train = df_train.withColumn("target",df_train["target"].cast(IntegerType()))
df_train = df_train.withColumn("ext_source_3", df_train["ext_source_3"].cast(DoubleType()))
df_train = df_train.withColumn("ext_source_2", df_train["ext_source_2"].cast(DoubleType()))
df_train = df_train.withColumn("name_education_type", df_train["name_education_type"].cast(StringType()))
df_train = df_train.withColumn("code_gender", df_train["code_gender"].cast(StringType()))
df_train = df_train.withColumn("amt_credit", df_train["amt_credit"].cast(DoubleType()))
df_train = df_train.withColumn("amt_income_total", df_train["amt_income_total"].cast(DoubleType()))
df_train = df_train.withColumn("flag_own_car", df_train["flag_own_car"].cast(StringType()))
df_train = df_train.withColumn("reg_city_not_work_city", df_train["reg_city_not_work_city"].cast(IntegerType()))
df_train = df_train.withColumn("flag_phone",df_train["flag_phone"].cast(IntegerType()))
df_train = df_train.withColumn("name_income_type",df_train["name_income_type"].cast(StringType()))
df_train = df_train.withColumn("name_family_status",df_train["name_family_status"].cast(StringType()))
df_train = df_train.withColumn("commonarea_medi",df_train["commonarea_medi"].cast(DoubleType()))
df_train = df_train.withColumn("commonarea_avg",df_train["commonarea_avg"].cast(DoubleType()))
df_train = df_train.withColumn("occupation_type",df_train["occupation_type"].cast(StringType()))


# Change dtypes column all_prev_app
df_prev_app = all_prev_app.select('sk_id_curr','amt_credit','amt_application')

# Change dtypes column all_prev_app
df_prev_app = df_prev_app.withColumn("sk_id_curr", df_prev_app["sk_id_curr"].cast(IntegerType()))
df_prev_app = df_prev_app.withColumn("amt_credit", df_prev_app["amt_credit"].cast(DoubleType()))
df_prev_app = df_prev_app.withColumn("amt_application", df_prev_app["amt_application"].cast(DoubleType()))

print((df_bureau.count(), len(df_bureau.columns)))
print((df_train.count(), len(df_train.columns)))
print((df_prev_app.count(), len(df_prev_app.columns)))

# Schema of Dataframe
df_bureau.printSchema()
df_train.printSchema()
df_prev_app.printSchema()


# # DATA AGGREGATION 

# categorical aggregation
previous_loan_counts = df_bureau.groupBy("sk_id_curr").agg(f.count("sk_id_bureau")).orderBy("sk_id_curr")
previous_loan_counts.show(10)
credit_active_counts = df_bureau.groupBy('sk_id_curr').agg(f.count("credit_active")).orderBy("sk_id_curr")
credit_active_counts.show(10)
name_family_status = df_train.groupBy('sk_id_curr').agg(f.count("name_family_status")).orderBy("sk_id_curr")
name_family_status.show(10)

# **Numeric Aggregating**
prev_agg = df_prev_app.groupBy("sk_id_curr").agg({'amt_credit': 'avg','amt_application' : 'avg'})
prev_agg.show(10)

bureau_agg = df_bureau.groupBy("sk_id_curr").agg({'amt_credit_sum': 'max'})
bureau_agg.show(10)

# Merging data
train = df_train.join(prev_agg, df_train.sk_id_curr == prev_agg.sk_id_curr).drop(prev_agg.sk_id_curr)
train.printSchema()
train.show()


# # DATA PROFILLING 
# Show dimension statistics
train.describe().show()

train.groupBy("target").count().show()
train.groupBy("code_gender").count().show()
train.groupBy("name_family_status").count().show()
train.groupBy("name_education_type").count().show() 
train.groupBy("name_income_type").count().show() 

print((train.count(), len(train.columns)))


# checking missing value
def na_value(df):
  columns = df.columns
  for col in columns :
    number_of_NA = df.filter(df[col].isNull()).count()
    number_of_rows = df.count()
    count_null = number_of_NA/number_of_rows
    print ('% of Null Value', col, ':', round(count_null*100,2),'%')

na_value(train)

# Drop columns null => 60 %
treshold = 0.6 
columns = train.columns
dropcol = []

def proportion_null(df):
  for col in columns :
    if (df.filter(df[col].isNull()).count()/df.count() >= treshold):
      dropcol.append(col)

proportion_null(train)
print('Feature dataframe will be removed = ', '\n', dropcol)

train_new = train.drop('commonarea_medi', 'commonarea_avg')
train_new.show(2)
train_new.printSchema()

na_value(train_new)


# Filling null value 
# train_new.fillna( { 'ext_source_3':med, 'ext_source_2':med }) #maintanance


# # SAMPLING DATA and CONVERT TO PANDAS

df_train = train_new.sample(False,0.1).toPandas()
print ('Number of rows and column df_train = ',df_train.shape)
df_train.head()
df_train.dtypes

# checking Null Value in pandas
def Null_Value(df):
  total = df.isnull().sum().sort_values(ascending = False)
  percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
  missing  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  return missing.head(15)

Null_Value(df_train)

# # FILLING MISSING VALUE
columns_null = ['ext_source_3','ext_source_2']
for col in columns_null:
  df_train[col] = df_train[col].fillna(df_train[col].median())

Null_Value(df_train)
  
  
# # EDA (EXploratory Data Anaysis)

# **Distribution of AMT_CREDIT**
plt.figure(figsize=(8,5))
sns.distplot(df_train["amt_credit"]).set_title('Distribution of AMT_CREDIT')

# **Distibution of AMT_INCOME_TOTAL**
plt.figure(figsize=(8,5))
sns.distplot(df_train["amt_income_total"]).set_title('Distribution of AMT_INCOME_TOTAL')

# **Data is balanced or imbalanced**
temp = df_train["target"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
plt.figure(figsize = (8,6))
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df).set_title('Application loans repayed - train dataset')

# **Family Status of Applicant's who applied for loan**
temp = df_train["name_family_status"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
plt.figure(figsize = (8,6))
sns.barplot(x = 'labels', y="values", data=df).set_title('Family Status')

# **Income sources of Applicant's who applied for loan**
temp = df_train["name_income_type"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
sns.barplot(x = 'labels', y="values", data=df).set_title('Income sources of Applicant')


# **Income sources of Applicant's who applied for loan**
temp = df_train["name_income_type"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values})
sns.barplot(x = 'labels', y="values", data=df).set_title('Income sources of Applicant')

# # ONE HOT-ENCODING
# Get dummy features
df_train = pd.get_dummies(df_train, drop_first=True)
df_train.shape
df_train.columns

# # BALANCING CLASS

# Split data to train and test
X = df_train.drop(['target'], axis=1)
y = df_train['target']
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


