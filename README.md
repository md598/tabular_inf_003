FastAPI Template 001


To run (with docker)
Clone the repo and use docker build . -t fastapi-fastai2 Then you can use docker run -p 8888:8000 fastapi-fastai2 and go to localhost:8888 to see the app.

Windows 10
When running docker run -p 8888:8000 fastapi-fastai2 after installation you may not be able to use localhost:8888. Docker was mapping ports to 127.0.0.1:8888 on my machine. Otherwise you can try netstat -a in your powershell to find the ip/port.

Thank you to Zach Mueller and his youtube series 'A walk with fastaiv2' on Tabular data and inference. And his Starlette Docker examples.
