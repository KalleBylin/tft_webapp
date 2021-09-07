FROM python:3.8

COPY ./app /app
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]