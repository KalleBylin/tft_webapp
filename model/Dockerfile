FROM tiangolo/uvicorn-gunicorn:python3.8

COPY ./app /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["python", "main.py"]
