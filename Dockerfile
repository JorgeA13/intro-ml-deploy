FROM python:3.10-slim-buster

WORKDIR /app

COPY api/requirements.txt /app

RUN pip3 install -r requirements.txt

COPY api/ ./api

COPY model/model.pkl ./model/model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]