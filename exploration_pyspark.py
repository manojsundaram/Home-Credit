# # Import pandas 
import pandas as pd
import numpy as np

# # Import Pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import first, collect_list, mean, countDistinct
from pyspark.sql import functions as F

# # Import Machine Learning Module 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report, confusion_matrix


# import packages for imbalance-learn for balancing class
from imblearn.over_sampling import SMOTE


# visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

# Create Spark Session
spark = SparkSession.builder.appName("Risk Model").getOrCreate()


# Import dataset
all_bureau = spark.table("poc_pengadaian.bureau")
all_train = spark.table("poc_pengadaian.application_train")
all_prev_app = spark.table("poc_pengadaian.previous_application")

type(all_train)

# Preview of Dataset
all_bureau.show(5)
all_train.show(5)
all_prev_app.show(5)

# # 4. CONVERT FROM RDD TO PANDAS
df= all_train.toPandas()


# # 5. EDA (EXploratory Data Anaysis)
# Descriptive of Statisics 

all_train.describe().show(2)



# # 6. DATA AGGREGATION OF SAMPLE_BUREAU

# **Count the number of previous loans and Credit Active** 
previous_loan_counts = all_bureau.groupBy("SK_ID_BUREAU").count()
previous_loan_counts.show(10)
credit_active_counts = all_bureau.groupBy('CREDIT_ACTIVE').count()
credit_active_counts.show(10)

# **Numeric Aggregating**





# # 5. DATA MERGING 


# # 4. CHECKING MISSING DATA
