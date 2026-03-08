from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/static", StaticFiles(directory="src"), name="static")
app.mount("/src", StaticFiles(directory="src"), name="src")
# -------- ENABLE CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- LOAD MODEL --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

model = joblib.load(model_path)

print("Model loaded from:", model_path)

# -------- INPUT FORMAT --------
class LandmarkInput(BaseModel):
    landmarks: list

# -------- SERVE UI --------
@app.get("/")
def home():
    file_path = os.path.join(os.path.dirname(__file__), "gesturetalk.html")
    return FileResponse(file_path)

# -------- PREDICT LETTER --------
@app.post("/predict-letter")
def predict_letter(data: LandmarkInput):

    landmarks = np.array(data.landmarks).reshape(1, -1)

    pred = model.predict(landmarks)[0]

    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(landmarks)))
    else:
        conf = 1.0

    return {
        "letter": pred,
        "confidence": conf
    }




























# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import joblib
# import numpy as np

# app = FastAPI()

# # -------- ENABLE CORS --------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------- LOAD MODEL --------
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

# model = joblib.load(model_path)

# print("Model loaded from:", model_path)

# # -------- INPUT FORMAT --------
# class LandmarkInput(BaseModel):
#     landmarks: list

# @app.get("/")
# def home():
#     return {"status": "Gesture API running"}

# # -------- PREDICT LETTER --------
# @app.post("/predict-letter")
# def predict_letter(data: LandmarkInput):

#     landmarks = np.array(data.landmarks).reshape(1, -1)

#     pred = model.predict(landmarks)[0]

#     if hasattr(model, "predict_proba"):
#         conf = float(np.max(model.predict_proba(landmarks)))
#     else:
#         conf = 1.0

#     return {
#         "letter": pred,
#         "confidence": conf
#     }














# from fastapi import FastAPI
# from pydantic import BaseModel
# import os
# import joblib
# import numpy as np

# app = FastAPI()

# # -------- LOAD MODEL --------
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

# model = joblib.load(model_path)


# # -------- INPUT FORMAT --------
# class LandmarkInput(BaseModel):
#     landmarks: list  # expects 63 values


# @app.get("/")
# def home():
#     return {"status": "Gesture API running"}


# # -------- PREDICT LETTER --------
# @app.post("/predict-letter")
# def predict_letter(data: LandmarkInput):
#     landmarks = np.array(data.landmarks).reshape(1, -1)
#     prediction = model.predict(landmarks)[0]
#     return {"letter": prediction}
