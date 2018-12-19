# # 1. IMPORT MODULE 
# **1.1. Import pandas** 
import pandas as pd
import numpy as np

# ** import utilites packages
import pickle
import cdsw

model = pickle.load(open("sklearn_rf.pkl","rb"))

def predict(args):
  df=np.array(args["feature"].split(",")).reshape(1,-1)
  columns = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT']
  
  
  df = pd.DataFrame(data=df,columns=columns,index=[0])
  
  df = df.astype({"FLAG_OWN_CAR": object
                  , "FLAG_OWN_REALTY": object
                  , "CNT_CHILDREN": np.int64
                  , "AMT_INCOME_TOTAL" : np.float64
                  , "AMT_CREDIT": np.float64
                  , "AMT_INCOME_TOTAL": np.float64
                 })
  
  account = pd.get_dummies(df, drop_first=False)
  
  return {"result" : model.predict(account)[0]}

#print(predict({"feature":"Y,Y,0,90000.0,454500.0"}))




