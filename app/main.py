# app/main.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU-only; prevents TF from trying to init CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # optional: reduce TF logs
import pickle
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras

# ---- CONFIG ----
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")  # set on Render or default /opt/model

# helper mapping
def map_or_unknown(key, mapping):
    key = str(key)
    return mapping.get(key, len(mapping))

# Request schema
class PredictRequest(BaseModel):
    soil: str
    sown: str
    soil_ph: float
    temp: float
    humidity: float
    N: float
    P: float
    K: float
    top_k: Optional[int] = 5

# Load model & artifacts on startup
model_path = os.path.join(MODEL_DIR, "model.keras")
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path} (set MODEL_DIR or put files there).")

model = keras.models.load_model(model_path)

# Load pickles
with open(os.path.join(MODEL_DIR, "scaler_x.pkl"), "rb") as f: scaler_x = pickle.load(f)
with open(os.path.join(MODEL_DIR, "dur_scaler.pkl"), "rb") as f: dur_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "wreq_scaler.pkl"), "rb") as f: wreq_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "le_name.pkl"), "rb") as f: le_name = pickle.load(f)
with open(os.path.join(MODEL_DIR, "le_water.pkl"), "rb") as f: le_water = pickle.load(f)
with open(os.path.join(MODEL_DIR, "soil_to_idx.pkl"), "rb") as f: soil_to_idx = pickle.load(f)
with open(os.path.join(MODEL_DIR, "sown_to_idx.pkl"), "rb") as f: sown_to_idx = pickle.load(f)

app = FastAPI(title="Crop Recommender")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        soil_idx = np.array([map_or_unknown(req.soil, soil_to_idx)], dtype=np.int32)
        sown_idx = np.array([map_or_unknown(req.sown, sown_to_idx)], dtype=np.int32)
        num_vec = scaler_x.transform(np.array([[req.soil_ph, req.temp, req.humidity, req.N, req.P, req.K]], dtype=float))

        preds = model.predict({"soil_in": soil_idx, "sown_in": sown_idx, "num_in": num_vec}, verbose=0)
        name_probs = preds[0][0]
        water_probs = preds[1][0]
        dur_scaled = float(preds[2].squeeze())
        wreq_scaled = float(preds[3].squeeze())
        growth_prob = float(preds[4].squeeze())

        name_idx = int(np.argmax(name_probs))
        water_idx = int(np.argmax(water_probs))
        name = le_name.inverse_transform([name_idx])[0]
        water = le_water.inverse_transform([water_idx])[0]
        dur = float(dur_scaler.inverse_transform([[dur_scaled]])[0,0])
        wreq = float(wreq_scaler.inverse_transform([[wreq_scaled]])[0,0]

)
        return {
            "predicted_name": name,
            "predicted_name_probs": name_probs.tolist(),
            "predicted_water": water,
            "predicted_water_probs": water_probs.tolist(),
            "predicted_duration": dur,
            "predicted_waterrequired": wreq,
            "predicted_growth_probability": float(growth_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
