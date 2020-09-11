from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import uvicorn
import asyncio
import aiohttp
import aiofiles
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
print(xgb.__version__)

# Run this in console:
# uvicorn example:app --reload --port 5000

#https://drive.google.com/uc?export=download&id=DRIVE_FILE_ID
# 9 Feature NN Model from 9_11_2020
#https://drive.google.com/file/d/1l66IevjsxMf3ONlgcxZJBTe9Rd6DPGm5/view?usp=sharing
classes = ['0','1','2','3','4','5','6','7','8','9','10','11']
path = Path(__file__).parent
export_file_url = '1l66IevjsxMf3ONlgcxZJBTe9Rd6DPGm5'
export_file_name = 'export.pkl'

tags_metadata = [
    {
        "name":"predict_nn",
        "description":"Predicted days until carrier possession. Will return 0 for same day, 1 for next day, 2 for two day, etc.",
    },

]
version = f"{sys.version_info.major}.{sys.version_info.minor}"
#path = Path(__file__).parent
app = FastAPI(openapi_tags=tags_metadata)

@app.on_event("startup")
def startup_event():
    #Need to redo with xgboost save... version control issues accross OSs
    xgb_open = open("app/models/xgb_model_9_11_2020__9_features.joblib.dat","rb") #docker version
    #xgb_open = open("models/xgb_model_9_11_2020__9_features.joblib.dat","rb") #local uvicorn
    global xgb_model
    xgb_model = joblib.load(xgb_open)
    gdd.download_file_from_google_drive(file_id=export_file_url, dest_path=path/export_file_name, unzip=True)
    try:
        learn = torch.load(path/export_file_name, map_location=torch.device('cpu'))
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
        print(e)

    df['Year']=df['order_time'].dt.year
    df['Month']=df['order_time'].dt.month
    df['Week']=df['order_time'].dt.week
    df['Day']=df['order_time'].dt.day
    df['Dayofweek']=df['order_time'].dt.dayofweek
    df['Dayofyear']=df['order_time'].dt.dayofyear
    df['Hour']=df['order_time'].dt.hour
    df['Elapsed']=df['order_time'].astype(np.int64) // 10 ** 9


@app.post("/predict_nn/", tags=["predict_nn"])
async def Predict_Days_To_Possession(data:List[Order]):
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
    xgb_preds = xgb_model.predict_proba(df.drop(['order_time','Year'],axis=1))
    argmax = xgb_preds.argmax(axis=1)
    predict=argmax#.numpy()
    response_body=predict
    lists = predict.tolist()
    json_str = json.dumps(lists)

    #nn_preds = learn.get_preds()[0]

    print (xgb_preds)
    print(predict)
    print("XGB Version:")
    print(xgb.__version__)
    return json_str
    #return response_json

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
