import json
import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult
from celery_worker import predict, celery_app, model
from TFT import TemporalFusionTransformer, params, BATCH_SIZE, DECODER_STEPS


class Inputs(BaseModel):
    inputs: List[List[float]] = []


app = FastAPI()

@app.get('/')
async def index():
    return {'message': 'online'}


@app.post('/predict')
async def predict_volatility(inputs: Inputs):
    results = inputs.dict()
    task = predict.delay(results["inputs"])

    response = {
            "task_id": task.id
        }
        
    return json.dumps(response)


@app.get('/predict/{task_id}')
async def predict_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)

    if task.ready():
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=8000)
