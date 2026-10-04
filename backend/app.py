from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
# import joblib

app = FastAPI()

class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float

# model = joblib.load("model.pkl")

@app.post("/predict")
def predict(req: PredictionRequest):
    input_array = np.array([[req.feature_1, req.feature_2]])
    prediction = 42
    return {
        "status": "ok",
        "prediction": prediction,
        "input": input_array.tolist()
    }
