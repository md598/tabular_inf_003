#FROM tiangolo/uvicorn-gunicorn:python3.7-alpine3.8
FROM tiangolo/uvicorn-gunicorn:python3.8-slim


LABEL maintainer="Sebastian Ramirez <tiangolo@gmail.com>"

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install numpy scipy #new
RUN pip install --upgrade pip

COPY requirements1.txt ./
RUN pip install --upgrade --no-cache-dir -r requirements1.txt

COPY requirements2.txt ./
RUN pip install --no-cache-dir -r requirements2.txt

RUN pip install --upgrade fastai

#COPY app app/
COPY . . 

RUN python app/main.py

EXPOSE 8000

#CMD ["python", "app/main.py", "main"]
