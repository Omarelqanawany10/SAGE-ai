from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# إنشاء FastAPI Application
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="🌾 Crop Prediction API",
    description="API للتنبؤ بنوع المحصول المناسب",
    version="1.0.0"
)

# ═══════════════════════════════════════════════════════════════
# تحميل الموديل والـ Pipeline
# ═══════════════════════════════════════════════════════════════

try:
    model = joblib.load('Rondom forest.pkl')
    pipeline = joblib.load('pipline.pkl')
    label_encoder = joblib.load('Label-Encoder.pkl')
    print("✅ تم تحميل الموديل بنجاح!")
except Exception as e:
    print(f"❌ خطأ: {str(e)}")
    raise

# ═══════════════════════════════════════════════════════════════
# تعريف الـ Input/Output
# ═══════════════════════════════════════════════════════════════

class CropFeatures(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.87,
                "humidity": 82.00,
                "ph": 6.50,
                "rainfall": 202.93
            }
        }

class PredictionResponse(BaseModel):
    predicted_crop: str

# ═══════════════════════════════════════════════════════════════
# Endpoint واحد بس للتنبؤ
# ═══════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictionResponse)
def predict_crop(features: CropFeatures):
    """
    التنبؤ بنوع المحصول
    """
    try:
        # تحويل لـ DataFrame
        input_df = pd.DataFrame([[
            features.N,
            features.P,
            features.K,
            features.temperature,
            features.humidity,
            features.ph,
            features.rainfall
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # تطبيق الـ pipeline
        processed = pipeline.transform(input_df)
        
        # التنبؤ
        prediction = model.predict(processed)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        # إرجاع اسم المحصول فقط
        return {"predicted_crop": crop_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server on http://localhost:8000")
    print("📝 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
