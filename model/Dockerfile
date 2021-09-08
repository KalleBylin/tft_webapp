FROM tiangolo/uvicorn-gunicorn:python3.8

COPY ./app /usr/src/app

RUN pip install -r /usr/src/app/requirements.txt
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

CMD python -u main.py 
