from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import numpy as np

app = FastAPI()

# -------- LOAD MODEL --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

model = joblib.load(model_path)


# -------- INPUT FORMAT --------
class LandmarkInput(BaseModel):
    landmarks: list  # expects 63 values


@app.get("/")
def home():
    return {"status": "Gesture API running"}


# -------- PREDICT LETTER --------
@app.post("/predict-letter")
def predict_letter(data: LandmarkInput):
    landmarks = np.array(data.landmarks).reshape(1, -1)
    prediction = model.predict(landmarks)[0]
    return {"letter": prediction}
