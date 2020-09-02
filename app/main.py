from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import uvicorn
import asyncio
import aiohttp
import aiofiles
import json

from typing import List
from pydantic import BaseModel,StrictFloat, validator, ValidationError
from datetime import date, datetime, time, timedelta, tzinfo
import sys
import pandas as pd
import numpy as np
import joblib

version = f"{sys.version_info.major}.{sys.version_info.minor}"
#path = Path(__file__).parent
app = FastAPI()

#Will enforce the correct data types getting to the model
class Data(BaseModel):
    input_order_time:datetime
    input_hours_truck:float

    @validator('input_order_time', pre=True, always=True)
    def set_ts_now(cls, v):
        return v or datetime.now()

#Need to redo with xgboost save... version control issues accross OSs  
#xgb_open = open("app/models/XGBoost_model_001.joblib.dat","rb")
#xgb_model = joblib.load(xgb_open)


@app.get("/")
def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn. Using Python {version}"
    return {"message": message}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict/")
async def predict(data:Data):
    order_time = data.input_order_time

    truck_hours = data.input_hours_truck


    return {"orig_value":order_time, "prediction":truck_hours-1}

@app.post("/predict2/")
async def predict2(data:List[Data]):
    response_body = []
    for item in data:
        order_time = item.input_order_time
        truck_hours = item.input_hours_truck


        response_body.append({"Input_Time":item.input_order_time,"Input_Truck":item.input_hours_truck,
        "Prediction":truck_hours-1,"Day":order_time.day,"Hour":order_time.hour,"Month":order_time.month,
        "Year":order_time.year})
    return response_body

def add_dateparts(df):
    try:
        df['order_time'] = pd.to_datetime(df['order_time'])
    except ValidationError as e:
        print(e)

    df['Year']=df['order_time'].dt.year
    df['Month']=df['order_time'].dt.month
    df['Week']=df['order_time'].dt.week
    df['Day']=df['order_time'].dt.day
    df['Dayofweek']=df['order_time'].dt.dayofweek
    df['Dayofyear']=df['order_time'].dt.dayofyear
    df['Hour']=df['order_time'].dt.hour
    df['Elapsed']=df['order_time'].astype(np.int64) // 10 ** 9

@app.post("/predict3/")
async def predict3(data:List[Data]):
    response_body = []
    rows_list=[]
    for item in data:
        order_time = item.input_order_time
        truck_hours = item.input_hours_truck
        if(order_time.hour < 15):
            is_before_3pm = True
            response_body="true"
        else:
            is_before_3pm = False
            response_body="false"
        rows_list.append({'order_time':order_time,'time_to_next_truck_bis_hours':truck_hours,'is_before_3pm':is_before_3pm})
    df=pd.DataFrame(rows_list)
    add_dateparts(df)

    df.head()
    df.to_csv('results.csv',index=False)
    df['order_time']=df['order_time'].dt.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')
    response_json= df.to_json(orient="records",date_format='iso')
    response_json= json.loads(response_json)
    #json.dumps(response_json,indent=4)
    #response_body= df.info()
    return response_json

@app.post("/predict4/")
async def predict4(data:List[Data]):
    rows_list=[]
    for item in data:
        order_time = item.input_order_time
        truck_hours = item.input_hours_truck
        if(order_time.hour < 15):
            is_before_3pm = True
        else:
            is_before_3pm = False
        rows_list.append({'order_time':order_time,'time_to_next_truck_bis_hours':truck_hours,'is_before_3pm':is_before_3pm})
    df=pd.DataFrame(rows_list)
    add_dateparts(df)

    df.head()
    df=df[['order_time','Year','Day', 'Dayofweek', 'Dayofyear', 'Month', 'Week', 'Hour', 'is_before_3pm', 'time_to_next_truck_bis_hours', 'Elapsed']]
    df.to_csv('results.csv',index=False)
    df['order_time']=df['order_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    response_json= df[['order_time','time_to_next_truck_bis_hours']].to_json(orient="records",date_format='iso')
    response_json= json.loads(response_json)
    #json.dumps(response_json,indent=4)
    response_body= df.info()
    
    #XGBoost predictions
    #xgb_preds = xgb_model.predict_proba(df.drop(['order_time','Year'],axis=1))
    #argmax = xgb_preds.argmax(axis=1)
    #predict=argmax#.numpy()
    #response_body=predict
    #lists = predict.tolist()
    #json_str = json.dumps(lists)
    
    
    #nn_preds = learn.get_preds()[0]
    
    #print (xgb_preds)
    #print(predict)

    return response_json


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
