from .models import PredictionRequest
from .app_utils import transform_df, get_model


model = get_model()


def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_df(request)
    prediction = model.predict(data_to_predict)[0]
    return max(0, prediction)