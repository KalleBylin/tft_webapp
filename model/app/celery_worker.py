import os
import json
import numpy as np
import torch
import celery
from celery import Celery
from TFT import TemporalFusionTransformer, params, BATCH_SIZE, DECODER_STEPS

celery_app = Celery('server', backend='redis://redis:6379/0', broker='redis://redis:6379/0')

model = TemporalFusionTransformer(params)
state_dict = torch.load('./checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


@celery_app.task(name="predict")
def predict(data):
    batch = {}
    batch["inputs"] = torch.from_numpy(np.array(data).reshape((BATCH_SIZE,DECODER_STEPS,9)))
    outputs, attention_weights = model(batch)
    outputs = outputs.cpu().detach().numpy()
    attn = attention_weights['multihead_attention'][0].cpu().detach().numpy().mean(axis=0)

    out = {"outputs": outputs.tolist(),
           "attention": attn.tolist()}

    return out