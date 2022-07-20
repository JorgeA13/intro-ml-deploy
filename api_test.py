import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_null_prediction():
    response = client.post('/v1/predict', json={'opening_gross': 0,
                                                'screens': 0,
                                                'production_budget': 0,
                                                'title_year': 0,
                                                'aspect_ratio': 0,
                                                'duration': 0,
                                                'cast_total_facebook_likes': 0,
                                                'budget': 0,
                                                'imdb_score': 0})
    assert response.status_code == 200
    assert response.json()['worldwide_gross'] == 0


def test_prediction():
    response = client.post('/v1/predict', json={'opening_gross': 23224,
                                                'screens': 323,
                                                'production_budget': 10345643,
                                                'title_year': 2020,
                                                'aspect_ratio': 123,
                                                'duration': 133,
                                                'cast_total_facebook_likes': 1244589,
                                                'budget': 11299473,
                                                'imdb_score': 5})
    assert response.status_code == 200
    assert response.json()['worldwide_gross'] != 0
