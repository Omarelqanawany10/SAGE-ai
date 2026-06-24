from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Precision Agriculture: Water Prediction API",
    description="Real-time irrigation prediction engine driven by a cross-validated HGB model.",
    version="1.0.0"
)

# 2. Load the trained pipeline into memory when the server boots
try:
    print("Loading ML engine into memory...")
    model_pipeline = joblib.load('crop_water_prediction_api_model.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")

# 3. Data Validation Schema (The Gatekeeper)
# This ensures that if a sensor sends a string instead of a float for Temperature, 
# the API cleanly rejects it instead of crashing the server.
class SensorData(BaseModel):
    CropType: str
    SoilType: str
    CropGrowthStage: str
    Season: str
    SoilpH: float
    SoilMoisture: float
    OrganicCarbon: float
    ElectricalConductivity: float
    TemperatureC: float
    Humidity: float
    Rainfallmm: float
    SunlightHours: float
    WindSpeedkmh: float
    Nitrogenkgha: float
    Phosphoruskgha: float
    Potassiumkgha: float
    IrrigationType: str
    MulchingUsed: str
    FieldAreahectare: float
    PreviousIrrigationmm: float

# 4. The Core Prediction Endpoint
@app.post("/predict")
async def predict_water(data: SensorData):
    try:
        # Convert the incoming verified JSON payload into a Pandas DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Pass the data through the Pipeline (Handles scaling, encoding, and predicting automatically)
        prediction = model_pipeline.predict(input_df)
        
        # Return a clean JSON response
        return {
            "status": "success",
            "predicted_water_need_mm_day": round(float(prediction[0]), 2),
            "message": "Calculated based on live sensor payload."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Engine Error: {str(e)}")

# 5. Health Check Endpoint
@app.get("/")
async def root():
    return {"status": "online", "message": "API is live and waiting for sensor data."}