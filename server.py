from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib, pandas as pd

app = FastAPI(title="SAGE API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Load Model 1: Crop classifier ──────────────────────────────
model_crop     = joblib.load("Rondom forest.pkl")
pipeline_crop  = joblib.load("pipline.pkl")
label_encoder  = joblib.load("Label-Encoder.pkl")

# ── Load Model 2: Water predictor ──────────────────────────────
model_water    = joblib.load("crop_water_prediction_api_model.pkl")


# ── Schemas ────────────────────────────────────────────────────
class CropFeatures(BaseModel):
    N: float; P: float; K: float
    temperature: float; humidity: float
    ph: float; rainfall: float

class WaterFeatures(BaseModel):
    CropType: str; SoilType: str; CropGrowthStage: str
    Season: str; SoilpH: float; SoilMoisture: float
    OrganicCarbon: float; ElectricalConductivity: float
    TemperatureC: float; Humidity: float; Rainfallmm: float
    SunlightHours: float; WindSpeedkmh: float
    Nitrogenkgha: float; Phosphoruskgha: float; Potassiumkgha: float
    IrrigationType: str; MulchingUsed: str
    FieldAreahectare: float; PreviousIrrigationmm: float


# ── Endpoints ──────────────────────────────────────────────────
@app.post("/crop/predict")
def predict_crop(f: CropFeatures):
    df = pd.DataFrame([f.model_dump()])
    processed = pipeline_crop.transform(df)
    pred = model_crop.predict(processed)
    return {"predicted_crop": label_encoder.inverse_transform(pred)[0]}

@app.post("/water/predict")
def predict_water(f: WaterFeatures):
    df = pd.DataFrame([f.model_dump()])
    mm_day = float(model_water.predict(df)[0])
    liters = round(mm_day * f.FieldAreahectare * 10000, 2)
    return {
        "water_need_mm_day": round(mm_day, 2),
        "water_need_liters_day": liters
    }

@app.get("/health")
def health(): return {"status": "online"}

# Serve the frontend (put index.html in a /static folder)
app.mount("/", StaticFiles(directory="static", html=True), name="static")