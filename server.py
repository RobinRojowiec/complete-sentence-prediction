"""

IDE: PyCharm
Project: complete-sentence-prediction
Author: Robin
Filename: server.py
Date: 18.01.2020

"""
import os
from json import dumps

import torch
import uvicorn
from fastapi import FastAPI
from simpletransformers.classification import ClassificationModel
from starlette.responses import Response
from torch.nn.functional import softmax

from static import LABELS

# load latest model
model = ClassificationModel('bert', 'outputs/', num_labels=2)

# initialize web app
app = FastAPI()


@app.get("/api/is_complete")
def read_root(text: str):
    predictions, raw_outputs = model.predict([text])
    tensor = torch.from_numpy(raw_outputs).float()
    probabilities = softmax(tensor, dim=1)

    result = []

    best = {"label": LABELS[predictions[0]], "confidence": probabilities[0][predictions[0]].item()}
    result.append(best)

    other_index = 1 if predictions[0] == 0 else 0
    other = {"label": LABELS[other_index], "confidence": probabilities[0][other_index].item()}
    result.append(other)

    return Response(content=dumps(result), media_type='application/json')


if __name__ == "__main__":
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)
