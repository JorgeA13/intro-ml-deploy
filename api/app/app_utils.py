from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO


def get_model() -> Pipeline:
    model_path = os.environ.get('MODEL_PATH', 'model/model.pkl')
    with open(model_path, 'rb') as model_file:
        model = load(BytesIO(model_file.read()))
    return model


def transform_df(model_data: BaseModel) -> DataFrame:
    dictionary_data = {key: [value] for key, value in model_data.dict().items()}
    data_df = DataFrame(dictionary_data)
    return data_df
