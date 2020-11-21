# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 03:05:48 2020

@author: Yllub-pc
"""

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from keras.models import load_model
import json 


def get_customer_merchant_relations():
    data_set=pd.read_csv(r'freq_dataset.csv')
    df=data_set.copy()

    customers=sorted(list(df['customer'].unique()))
    merchants=sorted(list(df['merchant'].unique()))
    
    data_set=(customers,merchants)
    
    return data_set

            
def load_trained_model() :
    model = load_model('trx_prediction_model.h5')
    return model

nn_model=load_trained_model()
customers=get_customer_merchant_relations()[0]
merchants=get_customer_merchant_relations()[1]


def get_prediction(customer_id):
    
     firstSeries=pd.Series(np.array([customers.index(customer_id)]*(len(merchants))))
     secondSeries=pd.Series(np.array(list(range(51))))
    
     predictions=nn_model.predict([firstSeries,secondSeries])
  
     merchant_frequency_map= {}
     
     for i in range(len(merchants)):
         merchant_frequency_map[merchants[i]]=str(predictions[i][0])
    
     return json.dumps(merchant_frequency_map,indent=4)












































