from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
#import nest_asyncio #Colab
#from pyngrok import ngrok #Colab
from fastapi.responses import HTMLResponse

import uvicorn
import asyncio
#import time
import json

from typing import List
from pydantic import BaseModel,StrictFloat, validator, ValidationError, Field
from datetime import date, datetime, time, timedelta, tzinfo
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from fastai.tabular.all import *
from google_drive_downloader import GoogleDriveDownloader as gdd

# Run this in console:
# uvicorn main:app --reload --port 5000

#https://drive.google.com/uc?export=download&id=DRIVE_FILE_ID
# 9 Feature NN Model from 9_11_2020
#https://drive.google.com/file/d/1l66IevjsxMf3ONlgcxZJBTe9Rd6DPGm5/view?usp=sharing

#classes = ['0','1','2','3','4','5','6','7','8','9','10','11'] # NN only
path = Path(__file__).parent
#path = !pwd # Colab
print("Current path: "),print(path)
export_file_url_NN = '1l66IevjsxMf3ONlgcxZJBTe9Rd6DPGm5'
export_file_url_xGB = '1-5nhr65OEL5dpqa-PQxuMFKOKxzQGcLf'
export_file_name_NN = 'export_NN.pkl'
export_file_name_XGB = 'export_XGB.pkl'
nn_path = str(path) + '/' + export_file_name_NN
XGB_path = str(path) + '/' + export_file_name_XGB
print(XGB_path)
print("XGB Version: "), print(xgb.__version__)
print("FastAI Version: "), print("2.0.12") #Sept 16th 2020

# Use this to not ensemble, but only use the NN
nn_only = False


# Json description for the docs page, we can attach it to a specific method
tags_metadata = [
    {
        "name":"predict",
        "description":"Predicted days until carrier possession. Will return 0 for same day, 1 for next day, 2 for two day, etc.",
    },
]
version = f"{sys.version_info.major}.{sys.version_info.minor}"
#path = Path(__file__).parent 
app = FastAPI(openapi_tags=tags_metadata)

#this is for things to work on Colab
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event("startup")
def startup_event():
    # Load in XGBoost
    gdd.download_file_from_google_drive(file_id=export_file_url_xGB, dest_path=XGB_path, unzip=True)
    xgb_open = open(XGB_path,"rb")
    global xgb_model
    xgb_model = joblib.load(xgb_open)

    # Load in the NN
    gdd.download_file_from_google_drive(file_id=export_file_url_NN, dest_path=nn_path, unzip=True)
    try:
        global learn
        learn = torch.load(nn_path, map_location=torch.device('cpu'))
        learn.dls.device = 'cpu'
        learn.to_fp32()
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


#Will enforce the correct data types getting to the model
class Order(BaseModel):
    input_order_time:datetime = Field(...,title='Time of order creation',alias='order_datetime',
    description='Time the order is placed. Must be a full datetime in the same timezone as the truck departure'\
    'must use ISO 8601 format - see https://www.w3.org/TR/NOTE-datetime',
    example='2020-07-31T23:43:27')

    input_hours_truck:float = Field(...,gt=0,lt=120,title='Time until the next truck departs',alias='hours_to_truck',
    description='Timedelta from when the order is placed until the trucks leave the fullfilment center, in hours.'\
    'i.e. the order is placed on 10pm Saturday and the next truck leaves at 530PM on Monday,'\
    ' then provide 43.5 hours. Use leading 0 if <1 )i.e. 0.5)',
    example='8.25')

# Need to test/improve the validator
    @validator('input_order_time', pre=True, always=True)
    def set_ts_now(cls, v):
        return v or datetime.now()


@app.get("/")
def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn. Using Python {version}"
    return {"message": message}


def add_dateparts(df):
    try:
        df['order_time'] = pd.to_datetime(df['order_time'])
    except ValidationError as e:
        raise print(e)

    df['Year']=df['order_time'].dt.year
    df['Month']=df['order_time'].dt.month
    df['Week']=df['order_time'].dt.week
    df['Day']=df['order_time'].dt.day
    df['Dayofweek']=df['order_time'].dt.dayofweek
    df['Dayofyear']=df['order_time'].dt.dayofyear
    df['Hour']=df['order_time'].dt.hour
    df['Elapsed']=df['order_time'].astype(np.int64) // 10 ** 9


@app.post("/predict/", tags=["predict"])
async def Predict_Days_To_Possession(data:List[Order]):
    rows_list=[]
    for item in data:
        order_time = item.input_order_time
        truck_hours = item.input_hours_truck
        if(order_time.hour < 15):
            is_before_3pm = True
        else:
            is_before_3pm = False
        rows_list.append({'order_time':order_time,'hours_to_next_truck':truck_hours,'is_before_3pm':is_before_3pm})
    df=pd.DataFrame(rows_list)
    add_dateparts(df)

    df.head()
    df=df[['order_time','Year','Day', 'Dayofweek', 'Dayofyear', 'Month', 'Week', 'Hour', 'is_before_3pm', 'hours_to_next_truck', 'Elapsed']]
    df.to_csv('results.csv',index=False)
    df['order_time']=df['order_time'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Simply resond with the inputs to test IO
    #response_json_io= df[['order_time','hours_to_next_truck']].to_json(orient="records",date_format='iso')
    #response_json_io= json.loads(response_json)
    print("Number of inputs:"+ str(len(df.index)))

    # NN Dataloader & batch
    dl = learn.dls.test_dl(df)#,bs=4, val_bs=4)
   
    # NN prediction
    inputs, nn_probs, _, nn_preds = learn.get_preds(dl=dl,with_input=True,with_decoded=True)
    print("nn Preds:")
    print(nn_preds)
    #response_json_nn = json.dumps(nn_preds.tolist()) # nn predictions only

    # XGBoost predictions
    xgb_probs = xgb_model.predict_proba(np.hstack((inputs[0].numpy(),inputs[1].numpy()))) #faster than making a new df
    xgb_preds = xgb_probs.argmax(axis=1)
    print("xgb_preds:")
    print(xgb_preds)
    #response_json_xgb = json.dumps(xgb_preds.tolist()) # xgb predictions only

    # Ensemble results
    avg = (nn_probs + xgb_probs) / 2
    ensemble_preds =avg.argmax(axis=1)
    print("Ensemble preds:")
    print(ensemble_preds)

    # Format final output
    df['predicted_days_to_posssession'] = ensemble_preds.tolist() #can change to preds from one model only if needed
    print("DF with preds")
    print(df)
    response_json= df[['order_time','hours_to_next_truck','predicted_days_to_posssession']].to_json(orient="records",date_format='iso')
    response_json= json.loads(response_json)

    #return response_json_io # Test IO by replying with input data
    return response_json

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
