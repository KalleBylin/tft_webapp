import json
import numpy as np
import torch
from celery import Celery
from celery.result import AsyncResult
from TFT import BATCH_SIZE, DECODER_STEPS

celery_app = Celery('server', backend='redis://redis:6379/0', broker='redis://redis:6379/0')

@celery_app.task
def predict(data):
    batch = {}
    batch["inputs"] = torch.from_numpy(np.array(data).reshape((BATCH_SIZE,DECODER_STEPS,9)))
    outputs, attention_weights = model(batch)
    outputs = outputs.cpu().detach().numpy()

    return json.dumps({"outputs": outputs.tolist()})