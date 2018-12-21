# # IMPORT MODULE 
# Import pandas** 
import pandas as pd
import numpy as np

# ** import utilites packages
import pickle
import cdsw

model = pickle.load(open("sklearn_rf_large.pkl","rb"))

# => Load model 
# _____________________________________________________

def predict(args):
  df=np.array(args["feature"].split(",")).reshape(1,-1)
  columns = ['NAME_INCOME_TYPE',
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
             'AMT_CREDIT']
  
# => create column yang akan dibuat unutk memprediksi
# _____________________________________________________
  
  df = pd.DataFrame(data=df,columns=columns,index=[0])
  

# => Ubah column tersebut kedalam bentuk dataframe dengan 
#    menggunakan pandas dataframe
# _____________________________________________________  
  
  
  df = df.astype({"NAME_INCOME_TYPE": object
                  ,"NAME_FAMILY_STATUS": object
                  ,"NAME_EDUCATION_TYPE": object
                  ,"REG_CITY_NOT_WORK_CITY" : np.int64
                  ,"FLAG_PHONE": np.int64
                  ,"FLAG_OWN_CAR": object
                  ,"CODE_GENDER" : object
                  ,"FLAG_OWN_REALTY" : object 
                  ,"EXT_SOURCE_3" : np.float64
                  ,"EXT_SOURCE_2" : np.float64
                  ,"AMT_INCOME_TOTAL" : np.float64
                  ,"AMT_CREDIT" : np.float64
                 })
  

# => ubah column tersebut kedalam data type yang sama
#   pada dataframe train model
# _____________________________________________________
  
  
  account = pd.get_dummies(df)
  

# => lakukan proses encode pada setiap type data kategorik  
# _____________________________________________________  
  
  
  col_dummies = ['REG_CITY_NOT_WORK_CITY', 'FLAG_PHONE', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
       'AMT_INCOME_TOTAL', 'AMT_CREDIT',
       'NAME_INCOME_TYPE_Commercial associate',
       'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner',
       'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
       'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
       'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated',
       'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Incomplete higher',
       'NAME_EDUCATION_TYPE_Lower secondary',
       'NAME_EDUCATION_TYPE_Secondary / secondary special', 'FLAG_OWN_CAR_Y',
       'CODE_GENDER_M', 'FLAG_OWN_REALTY_Y']
  
# => Ambil semua feature yang sudah dilakukan encode pada proses train
#    tujuan nya adalah untuk dilakukan transform antara column train dan predict
#    memiliki dimensi yang sama. 
#___________________________________________________________________________________


  feature = df.T.reindex(col_dummies).T.fillna(0)

# => code diatas digunakan untuk melakukan transform columns 
  
  
  return {"result" : model.predict(feature)[0]}

#result = predict({"feature":"Working,Married,Secondary / secondary special,2,0,Y,M,Y,39203.0,32032.0,4849020.0,49024.0"})
 
#print(result)



