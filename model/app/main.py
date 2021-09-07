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
from TFT import TemporalFusionTransformer, params, BATCH_SIZE, DECODER_STEPS


celery_app = Celery('tasks', backend='redis://redis:6379/0', broker='redis://redis:6379/0')


model = TemporalFusionTransformer(params)
state_dict = torch.load('./checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


@celery_app.task
def predict(data):
    batch = {}
    batch["inputs"] = torch.from_numpy(np.array(data).reshape((BATCH_SIZE,DECODER_STEPS,9)))
    outputs, attention_weights = model(batch)
    outputs = outputs.cpu().detach().numpy()

    return json.dumps({"outputs": outputs.tolist()})


class Inputs(BaseModel):
    inputs: List[List[float]] = []


app = FastAPI()

@app.get('/')
def index():
    return {'message': 'online'}


@app.post('/predict')
def predict_volatility(inputs: Inputs):
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
        print(task.result)
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
