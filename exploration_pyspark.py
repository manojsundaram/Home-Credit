# # Import pandas 
import pandas as pd
import numpy as np

# # Import Pyspark
from pyspark.sql import SparkSession


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

# Preview of Dataset
all_bureau.head()
all_train.head()
all_prev_app.head()

# # 3. EDA (EXploratory Data Anaysis)
all_bureau.describe()
all_train.describe()
all_prev_app.describe()





