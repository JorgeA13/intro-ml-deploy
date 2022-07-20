from fastapi import FastAPI

from .app.models import PredictionResponse, PredictionRequest
from .app.views import get_prediction


app = FastAPI(docs_url='/')


@app.post('/v1/predict')
def make_prediction(request: PredictionRequest):
    return PredictionResponse(worldwide_gross=get_prediction(request))
