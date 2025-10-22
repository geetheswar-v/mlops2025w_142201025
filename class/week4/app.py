from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pydantic import BaseModel

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_data = joblib.load('model.joblib')
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model file not found. Please run train.py first.")
        model_data = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model_data = None

    app.state.model_data = model_data
    yield
    
    print("Shutting down application...")
    
app = FastAPI(
    title="Iris Classification API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "Iris Classification API",
        "version": "1.0.0"
    }


@app.post("/predict")
async def predict(iris_features: IrisFeatures):
    model_data = app.state.model_data
    if model_data is None:
        return {"error": "Model not loaded. Please run train.py first."}
    
    features = np.array([
        iris_features.sepal_length,
        iris_features.sepal_width,
        iris_features.petal_length,
        iris_features.petal_width
    ]).reshape(1, -1)
    
    prediction = model_data['model'].predict(features)[0]
    prediction_proba = model_data['model'].predict_proba(features)[0]
    
    species = model_data['target_names'][prediction]
    
    return {
        "prediction": int(prediction),
        "species": species,
        "probabilities": {
            model_data['target_names'][i]: float(prob) 
            for i, prob in enumerate(prediction_proba)
        }
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)